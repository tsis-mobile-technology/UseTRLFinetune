#!/usr/bin/env python3
"""
기존 파인튜닝 모델에서 추가 학습을 진행하는 스크립트
두 가지 방식 지원:
1. 기존 LoRA 어댑터에서 바로 계속 학습 (incremental training)
2. 병합된 모델에서 새 LoRA 어댑터로 학습 (merged model training)
"""

import os
import argparse
import logging
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('continue_train')

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='기존 파인튜닝 모델에서 추가 학습')
    
    # 기본 인자
    parser.add_argument('--base-model', default='EleutherAI/polyglot-ko-1.3b', 
                        help='기본 모델 이름 또는 경로')
    parser.add_argument('--finetuned-model', default=None,
                        help='기존에 파인튜닝된 모델 경로 (my_optimized_model 또는 my_korean_finetuned_model)')
    parser.add_argument('--data-path', required=True, 
                        help='새로운 훈련 데이터 JSONL 파일 경로')
    parser.add_argument('--output-dir', default='my_continued_model', 
                        help='모델 저장 디렉토리')
    
    # 학습 방식 설정
    parser.add_argument('--mode', choices=['continue', 'merge_and_new'], default='continue',
                        help='학습 방식: continue(기존 어댑터 계속 학습) 또는 merge_and_new(병합 후 새 어댑터)')
    parser.add_argument('--merge-and-save', action='store_true',
                        help='학습 후 어댑터와 모델 병합해서 저장 (병합된 전체 모델)')
    
    # 학습 하이퍼파라미터
    parser.add_argument('--batch-size', type=int, default=4, 
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='훈련 에포크 수')
    parser.add_argument('--learning-rate', type=float, default=2e-4, 
                        help='학습률')
    parser.add_argument('--max-length', type=int, default=512, 
                        help='최대 시퀀스 길이')
    parser.add_argument('--limit-samples', type=int, default=1000, 
                        help='훈련에 사용할 최대 샘플 수 (메모리 제한)')
    
    # LoRA 설정
    parser.add_argument('--lora-r', type=int, default=64,
                        help='LoRA r 값 (rank)')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha 값')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='LoRA dropout 비율')
    
    # 하드웨어 최적화 설정
    parser.add_argument('--use-8bit', action='store_true',
                        help='8비트 양자화 사용')
    parser.add_argument('--use-4bit', action='store_true',
                        help='4비트 양자화 사용 (더 적은 메모리 사용)')
    parser.add_argument('--use-fp16', action='store_true',
                        help='FP16 훈련 사용')
    
    return parser.parse_args()

def build_dataset_from_jsonl(data_path, tokenizer, max_length=512, limit=None):
    """
    JSONL 파일에서 데이터셋 구축
    """
    logger.info(f"JSONL 파일에서 데이터셋 구축: {data_path}")
    
    # JSONL 파일 로드
    ds = load_dataset('json', data_files=data_path, split='train')
    
    # 필요한 경우 샘플 수 제한
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
        logger.info(f"데이터셋을 {limit}개 샘플로 제한")
    
    # 텍스트 필드 확인
    if 'text' not in ds.column_names:
        logger.warning("데이터셋에 'text' 필드가 없습니다. 열 이름을 확인합니다.")
        logger.info(f"사용 가능한 열: {ds.column_names}")
        
        # 가능한 텍스트 필드 이름들
        possible_text_fields = ['text', 'content', 'dialogue', 'conversation', 'message', 'document']
        
        text_field = None
        for field in possible_text_fields:
            if field in ds.column_names:
                text_field = field
                logger.info(f"'{field}' 필드를 텍스트로 사용합니다.")
                break
        
        if text_field is None:
            # 첫 번째 열을 텍스트 필드로 사용
            text_field = ds.column_names[0]
            logger.warning(f"텍스트 필드를 찾을 수 없어 첫 번째 열 '{text_field}'을 사용합니다.")
        
        # 열 이름 변경
        ds = ds.rename_column(text_field, 'text')
    
    # 텍스트 길이 필터링
    ds = ds.filter(lambda x: len(x["text"]) > 20, batched=False)
    logger.info(f"필터링 후 데이터셋 크기: {len(ds)}")
    
    def tokenize_function(examples):
        # 텍스트 토큰화
        tokenized = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        )
        # labels는 input_ids와 동일하게 설정 (언어모델링)
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # 토큰화 적용
    tokenized_ds = ds.map(
        tokenize_function, 
        batched=True, 
        remove_columns=ds.column_names
    )
    
    logger.info(f"데이터셋 구축 완료: {len(tokenized_ds)}개 샘플")
    return tokenized_ds

def load_base_model(args):
    """기본 모델 로드 (양자화 옵션 적용)"""
    logger.info(f"기본 모델 로드: {args.base_model}")
    
    # 양자화 설정
    quantization_config = None
    if args.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    elif args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16 if args.use_fp16 else None,
        trust_remote_code=True
    )
    
    return model

def detect_target_modules(model):
    """모델에서 LoRA 타겟 모듈 자동 감지"""
    logger.info("LoRA 타겟 모듈 감지 중...")
    
    # 일반적인 타겟 모듈 패턴
    common_patterns = [
        "query_key_value", "dense_h_to_4h", "dense_4h_to_h", "dense",  # Polyglot
        "c_attn", "c_proj", "c_fc", "c_proj",  # GPT 스타일
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",  # LLaMA 스타일
        "k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2",  # BERT 스타일
        "attention.self", "attention.output.dense",  # 다른 스타일
        "linear", "Linear"  # 일반적인 이름
    ]
    
    # 모델에 실제로 존재하는 모듈 찾기
    found_targets = set()
    for name, _ in model.named_modules():
        for pattern in common_patterns:
            if pattern in name.split("."):
                module_name = name.split(".")[-1]
                found_targets.add(module_name)
    
    # 결과 정리
    target_modules = list(found_targets)
    
    # 타겟 모듈이 없으면 모든 선형 레이어 찾기
    if not target_modules:
        logger.warning("일반적인 타겟 모듈을 찾을 수 없습니다. 모든 Linear 레이어를 탐색합니다.")
        
        import torch.nn as nn
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module_name = name.split('.')[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)
    
    # 최소 타겟 모듈 보장
    if not target_modules:
        logger.warning("타겟 모듈을 찾을 수 없어 기본값 'dense'를 사용합니다.")
        target_modules = ["dense"]
    
    logger.info(f"감지된 LoRA 타겟 모듈: {target_modules}")
    return target_modules

def continue_training(args):
    """기존 LoRA 어댑터에서 계속 학습하는 함수"""
    logger.info(f"모드: 기존 어댑터에서 계속 학습")
    
    # 기본 모델 및 토크나이저 로드
    model = load_base_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 양자화된 모델인 경우 k-bit 학습을 위해 전처리
    if args.use_8bit or args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # 기존 파인튜닝된 어댑터 로드
    if args.finetuned_model:
        logger.info(f"기존 파인튜닝된 어댑터 로드: {args.finetuned_model}")
        model = PeftModel.from_pretrained(model, args.finetuned_model)
        logger.info("기존 어댑터 로드 완료")
    else:
        # 타겟 모듈 감지
        target_modules = detect_target_modules(model)
        
        # 새 LoRA 설정
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        # PEFT 모델 생성
        model = get_peft_model(model, lora_config)
        logger.info("새 LoRA 어댑터 초기화 완료")
    
    # 학습 가능한 파라미터 정보 출력
    model.print_trainable_parameters()
    
    # 데이터셋 구축
    dataset = build_dataset_from_jsonl(
        args.data_path, 
        tokenizer,
        args.max_length,
        args.limit_samples
    )
    
    # 데이터 콜레이터 (언어 모델링용)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # 인과적 언어 모델링
    )
    
    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=args.learning_rate,
        fp16=args.use_fp16,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    # 훈련 실행
    logger.info("훈련 시작...")
    trainer.train()
    
    # 모델 저장
    logger.info(f"훈련된 모델 저장: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 훈련 설정 저장
    training_config = {
        "model_name": args.base_model,
        "finetuned_from": args.finetuned_model,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, 
        "lora_dropout": args.lora_dropout,
        "training_mode": "continue",
        "optimizations": {
            "use_8bit": args.use_8bit,
            "use_4bit": args.use_4bit,
            "fp16": args.use_fp16
        }
    }
    
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    
    # 병합 및 저장이 필요한 경우
    if args.merge_and_save:
        merge_and_save_model(model, tokenizer, args)
    
    logger.info("모델 파인튜닝 완료")
    return model, tokenizer

def merge_and_train_new(args):
    """
    기존 어댑터를 기본 모델과 병합한 후 새 어댑터로 학습하는 함수
    """
    logger.info(f"모드: 기존 어댑터 병합 후 새 어댑터 학습")
    
    if not args.finetuned_model:
        logger.error("병합 및 새 학습 모드에는 --finetuned-model 인자가 필요합니다.")
        return None, None
    
    # 양자화 없이 기본 모델 로드 (병합을 위해)
    logger.info(f"기본 모델 로드: {args.base_model}")
    
    temp_args = argparse.Namespace(**vars(args))
    temp_args.use_8bit = False
    temp_args.use_4bit = False
    model = load_base_model(temp_args)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 기존 파인튜닝된 어댑터 로드
    logger.info(f"기존 파인튜닝된 어댑터 로드: {args.finetuned_model}")
    model = PeftModel.from_pretrained(model, args.finetuned_model)
    logger.info("기존 어댑터 로드 완료")
    
    # 어댑터 병합
    logger.info("어댑터를 기본 모델과 병합 중...")
    merged_model = model.merge_and_unload()
    logger.info("병합 완료")
    
    # 병합된 모델 임시 저장
    temp_merged_dir = os.path.join(args.output_dir, "temp_merged")
    os.makedirs(temp_merged_dir, exist_ok=True)
    merged_model.save_pretrained(temp_merged_dir)
    tokenizer.save_pretrained(temp_merged_dir)
    logger.info(f"병합된 모델 임시 저장: {temp_merged_dir}")
    
    # 메모리 정리
    del model
    del merged_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 병합된 모델을 기본 모델로 설정하여 새 학습 진행
    logger.info("병합된 모델을 기반으로 새 LoRA 어댑터 학습 시작")
    new_args = argparse.Namespace(**vars(args))
    new_args.base_model = temp_merged_dir
    new_args.finetuned_model = None  # 새 어댑터 학습
    
    # 새 학습 실행
    model, tokenizer = continue_training(new_args)
    
    # 임시 디렉토리 정리 (선택적)
    # import shutil
    # shutil.rmtree(temp_merged_dir)
    
    return model, tokenizer

def merge_and_save_model(model, tokenizer, args):
    """
    LoRA 어댑터와 기본 모델을 병합하여 완전한 모델로 저장
    """
    logger.info("어댑터와 기본 모델 병합 중...")
    
    try:
        # 어댑터 병합
        merged_model = model.merge_and_unload()
        
        # 병합된 모델 저장
        merged_dir = os.path.join(args.output_dir, "merged_model")
        os.makedirs(merged_dir, exist_ok=True)
        
        logger.info(f"병합된 모델 저장: {merged_dir}")
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        
        # 메모리 정리
        del merged_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("모델 병합 및 저장 완료")
        return True
    
    except Exception as e:
        logger.error(f"모델 병합 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    args = parse_args()
    
    # 데이터 파일 확인
    if not os.path.exists(args.data_path):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {args.data_path}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 학습 방식에 따라 다른 함수 호출
    try:
        if args.mode == 'continue':
            continue_training(args)
        elif args.mode == 'merge_and_new':
            merge_and_train_new(args)
        else:
            logger.error(f"지원하지 않는 모드: {args.mode}")
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()