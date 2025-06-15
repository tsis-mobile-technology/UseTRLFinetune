"""
PPOTrainer API 문제를 피하기 위한 더 간단한 커스텀 파인튜닝 스크립트
"""
import os
import argparse
import logging
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simple_train')

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='간단한 커스텀 데이터로 한국어 언어 모델 파인튜닝')
    
    parser.add_argument('--model-name', default='EleutherAI/polyglot-ko-1.3b', 
                        help='파인튜닝할 모델 이름')
    parser.add_argument('--data-path', required=True, 
                        help='훈련 데이터 JSONL 파일 경로')
    parser.add_argument('--output-dir', default='my_korean_finetuned_model', 
                        help='모델 저장 디렉토리')
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
        raise ValueError("데이터셋에 'text' 필드가 없습니다.")
    
    # 텍스트 길이 필터링
    ds = ds.filter(lambda x: len(x["text"]) > 50, batched=False)
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

def train_with_simple_method(args):
    """
    간단한 Hugging Face Trainer를 사용한 파인튜닝
    """
    logger.info("모델 파인튜닝 시작 (Hugging Face Trainer 사용)")
    
    # 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    # 모델 및 토크나이저 로드
    logger.info(f"모델 로드: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # k-bit 학습을 위해 모델을 전처리
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정 - polyglot-ko 모델에 맞는 타겟 모듈 찾기
    logger.info("모델 구조 분석 중...")
    
    # 모델의 모든 모듈 이름을 출력하여 적절한 타겟 찾기
    target_modules = []
    for name, module in model.named_modules():
        if 'linear' in name.lower() or 'attention' in name.lower():
            logger.info(f"잠재적 타겟 모듈: {name}")
            if len(target_modules) < 10:  # 처음 10개만 저장
                target_modules.append(name.split('.')[-1])  # 마지막 부분만 저장
    
    # polyglot 모델에 일반적으로 사용되는 타겟 모듈들
    common_targets = ["c_attn", "c_proj", "c_fc", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    
    # 실제 존재하는 타겟 모듈 찾기
    valid_targets = []
    for target in common_targets:
        for name, _ in model.named_modules():
            if target in name:
                valid_targets.append(target)
                break
    
    # 타겟 모듈이 없으면 기본값 사용
    if not valid_targets:
        # 모든 Linear 레이어를 타겟으로 설정
        import torch.nn as nn
        valid_targets = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module_name = name.split('.')[-1]
                if module_name not in valid_targets:
                    valid_targets.append(module_name)
                if len(valid_targets) >= 4:  # 최대 4개까지만
                    break
    
    logger.info(f"사용할 타겟 모듈: {valid_targets}")
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=valid_targets if valid_targets else ["c_attn"]  # 기본값으로 fallback
    )
    
    # PEFT 모델 생성
    model = get_peft_model(model, lora_config)
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
        fp16=True,  # 메모리 절약을 위한 혼합 정밀도
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
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
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 훈련 설정 저장
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    logger.info("모델 파인튜닝 완료")

def main():
    """메인 함수"""
    args = parse_args()
    
    # 데이터 파일 확인
    if not os.path.exists(args.data_path):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {args.data_path}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 훈련
    try:
        train_with_simple_method(args)
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()