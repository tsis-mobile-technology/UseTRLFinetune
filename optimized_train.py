"""
RTX 3060 12GB 최적화된 한국어 언어 모델 파인튜닝 스크립트
"""
import os
import argparse
import logging
import json
import torch
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments,
    AutoModelForCausalLM, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('optimized_train')

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='RTX 3060 12GB 최적화된 한국어 언어 모델 파인튜닝')
    
    parser.add_argument('--model-name', default='EleutherAI/polyglot-ko-1.3b', 
                        help='파인튜닝할 모델 이름')
    parser.add_argument('--data-path', required=True, 
                        help='훈련 데이터 JSONL 파일 경로')
    parser.add_argument('--output-dir', default='my_korean_finetuned_model', 
                        help='모델 저장 디렉토리')
    
    # 최적화된 기본값들
    parser.add_argument('--batch-size', type=int, default=8, 
                        help='배치 크기 (12GB GPU 최적화)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, 
                        help='그래디언트 누적 스텝 (효과적 배치 크기 = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='훈련 에포크 수')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                        help='학습률')
    parser.add_argument('--max-length', type=int, default=2048, 
                        help='최대 시퀀스 길이 (12GB GPU 최적화)')
    parser.add_argument('--limit-samples', type=int, default=None, 
                        help='훈련에 사용할 최대 샘플 수')
    
    # LoRA 최적화 파라미터
    parser.add_argument('--lora-r', type=int, default=128, 
                        help='LoRA rank (높을수록 더 많은 파라미터 훈련)')
    parser.add_argument('--lora-alpha', type=int, default=256, 
                        help='LoRA alpha (일반적으로 rank의 2배)')
    parser.add_argument('--lora-dropout', type=float, default=0.05, 
                        help='LoRA dropout')
    
    # 고급 옵션
    parser.add_argument('--use-4bit', action='store_true', 
                        help='4bit 양자화 사용 (더 많은 메모리 절약)')
    parser.add_argument('--use-flash-attention', action='store_true', 
                        help='Flash Attention 사용 (더 빠른 훈련)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, 
                        help='Warmup 비율')
    parser.add_argument('--weight-decay', type=float, default=0.01, 
                        help='Weight decay')
    parser.add_argument('--lr-scheduler', choices=['cosine', 'linear', 'constant'], 
                        default='cosine', help='학습률 스케줄러')
    
    return parser.parse_args()

def get_optimized_quantization_config(use_4bit=False):
    """최적화된 양자화 설정"""
    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )

def build_optimized_dataset(data_path, tokenizer, max_length=2048, limit=None):
    """
    최적화된 데이터셋 구축
    """
    logger.info(f"최적화된 데이터셋 구축: {data_path}")
    
    # JSONL 파일 로드
    ds = load_dataset('json', data_files=data_path, split='train')
    
    # 필요한 경우 샘플 수 제한
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
        logger.info(f"데이터셋을 {limit}개 샘플로 제한")
    
    # 텍스트 필드 확인
    if 'text' not in ds.column_names:
        raise ValueError("데이터셋에 'text' 필드가 없습니다.")
    
    # 텍스트 길이 필터링 (더 엄격한 필터링)
    def filter_by_length(example):
        text_length = len(example["text"])
        return 100 <= text_length <= max_length * 2  # 토큰 수는 대략 문자 수의 절반
    
    ds = ds.filter(filter_by_length, batched=False)
    logger.info(f"길이 필터링 후 데이터셋 크기: {len(ds)}")
    
    # 텍스트 품질 필터링
    def filter_by_quality(example):
        text = example["text"]
        # 기본적인 품질 필터
        if len(text.split()) < 20:  # 너무 짧은 텍스트
            return False
        if text.count('\n') / len(text) > 0.1:  # 너무 많은 줄바꿈
            return False
        return True
    
    ds = ds.filter(filter_by_quality, batched=False)
    logger.info(f"품질 필터링 후 데이터셋 크기: {len(ds)}")
    
    def tokenize_function(examples):
        # 정적 패딩 사용 (안정성 우선)
        tokenized = tokenizer(
            examples["text"], 
            padding="max_length",  # 정적 패딩으로 변경
            truncation=True, 
            max_length=max_length,
            return_tensors=None  # 텐서 변환 비활성화 (datasets가 처리)
        )
        # labels는 input_ids와 동일하게 설정
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # 토큰화 적용 (안정성을 위해 단일 프로세스 사용)
    tokenized_ds = ds.map(
        tokenize_function, 
        batched=True,
        batch_size=16,  # 토큰화 시 적당한 배치 사용
        remove_columns=ds.column_names,
        num_proc=1  # 안정성을 위해 단일 프로세스 사용
    )
    
    logger.info(f"최적화된 데이터셋 구축 완료: {len(tokenized_ds)}개 샘플")
    return tokenized_ds

def get_optimized_lora_config(args, model):
    """모델에 맞는 최적화된 LoRA 설정"""
    
    # 모델의 모든 모듈 이름 출력 및 분석
    logger.info("모델 구조 분석 중...")
    target_modules = set()
    
    # 모든 Linear 레이어 찾기
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 모듈 이름의 마지막 부분 추출
            module_name = name.split('.')[-1]
            target_modules.add(module_name)
            logger.info(f"Linear 모듈 발견: {name} -> {module_name}")
    
    # target_modules를 리스트로 변환
    target_modules = list(target_modules)
    
    if not target_modules:
        # 백업: 일반적인 Transformer 모듈들
        logger.warning("Linear 모듈을 찾지 못했습니다. 기본 모듈들을 시도합니다.")
        # 모든 모듈 이름 출력
        all_modules = []
        for name, module in model.named_modules():
            all_modules.append(name)
            if len(all_modules) < 20:  # 처음 20개만 출력
                logger.info(f"모듈: {name} -> {type(module).__name__}")
        
        # polyglot 모델에서 자주 사용되는 패턴들
        possible_targets = []
        for name, _ in model.named_modules():
            if any(pattern in name for pattern in ['query', 'key', 'value', 'dense', 'proj', 'fc', 'attn']):
                module_name = name.split('.')[-1]
                if module_name not in possible_targets:
                    possible_targets.append(module_name)
                    logger.info(f"가능한 타겟: {name} -> {module_name}")
        
        if possible_targets:
            target_modules = possible_targets[:4]  # 최대 4개만 선택
        else:
            # 최후의 수단: 모든 모듈 검사
            target_modules = ["dense", "query", "key", "value"]  # 일반적인 이름들
    
    logger.info(f"최종 선택된 LoRA 타겟 모듈: {target_modules}")
    
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        # 추가 최적화 옵션
        modules_to_save=None,  # 모든 모듈을 LoRA로 처리
    )

class OptimizedDataCollator:
    """최적화된 데이터 콜레이터"""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # 기본 DataCollatorForLanguageModeling 사용
        from transformers import DataCollatorForLanguageModeling
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 인과적 언어 모델링
        )
        return collator(features)

def train_with_optimized_method(args):
    """
    최적화된 훈련 방법
    """
    logger.info(f"RTX 3060 12GB 최적화 훈련 시작")
    logger.info(f"배치 크기: {args.batch_size}, 그래디언트 누적: {args.gradient_accumulation_steps}")
    logger.info(f"효과적 배치 크기: {args.batch_size * args.gradient_accumulation_steps}")
    
    # 양자화 설정
    quantization_config = get_optimized_quantization_config(args.use_4bit)
    
    # 모델 로드 (최적화된 설정)
    logger.info(f"모델 로드: {args.model_name}")
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,  # 더 나은 수치 안정성
    }
    
    # Flash Attention 지원 확인
    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Flash Attention 2 사용")
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # k-bit 학습 준비
    model = prepare_model_for_kbit_training(model)
    
    # 최적화된 LoRA 설정
    lora_config = get_optimized_lora_config(args, model)
    model = get_peft_model(model, lora_config)
    
    # 훈련 가능한 파라미터 출력
    model.print_trainable_parameters()
    
    # 최적화된 데이터셋 구축
    dataset = build_optimized_dataset(
        args.data_path, 
        tokenizer,
        args.max_length,
        args.limit_samples
    )
    
    # 최적화된 데이터 콜레이터
    data_collator = OptimizedDataCollator(tokenizer, args.max_length)
    
    # 총 훈련 스텝 계산
    total_steps = len(dataset) * args.epochs // (args.batch_size * args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    logger.info(f"총 훈련 스텝: {total_steps}, Warmup 스텝: {warmup_steps}")
    
    # 최적화된 훈련 인자
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # 학습률 및 옵티마이저 설정
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        
        # 메모리 최적화
        fp16=False,
        bf16=True,  # BF16 사용 (더 안정적)
        dataloader_pin_memory=True,
        gradient_checkpointing=True,  # 메모리 절약
        
        # 로깅 및 저장
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        
        # 추가 최적화
        remove_unused_columns=False,
        report_to=None,  # wandb 등 사용하지 않음
        prediction_loss_only=True,
        
        # 성능 최적화
        dataloader_num_workers=4,
    )
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # 훈련 실행
    logger.info("최적화된 훈련 시작...")
    trainer.train()
    
    # 모델 저장
    logger.info(f"훈련된 모델 저장: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 훈련 설정 저장
    training_config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        'optimizations': {
            'use_4bit': args.use_4bit,
            'use_flash_attention': args.use_flash_attention,
            'gradient_checkpointing': True,
            'bf16': True,
            'group_by_length': True,
        }
    }
    
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)
    
    logger.info("최적화된 모델 파인튜닝 완료!")
    
    # 메모리 정리
    torch.cuda.empty_cache()

def main():
    """메인 함수"""
    args = parse_args()
    
    # 데이터 파일 확인
    if not os.path.exists(args.data_path):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {args.data_path}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # GPU 메모리 정보 출력
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU 메모리: {gpu_memory:.1f}GB")
        
        # RTX 3060 12GB 최적화 권장사항
        if gpu_memory > 11:
            logger.info("RTX 3060 12GB 감지 - 최적화된 설정 사용")
            if not args.use_4bit and args.max_length > 2048:
                logger.warning("긴 시퀀스 길이 사용 시 --use-4bit 옵션 권장")
    
    # 모델 훈련
    try:
        train_with_optimized_method(args)
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 메모리 부족 시 권장사항
        if "CUDA out of memory" in str(e):
            logger.error("메모리 부족! 다음 옵션들을 시도해보세요:")
            logger.error("1. --batch-size 줄이기 (예: --batch-size 4)")
            logger.error("2. --max-length 줄이기 (예: --max-length 1024)")
            logger.error("3. --use-4bit 옵션 사용")
            logger.error("4. --gradient-accumulation-steps 늘리기")

if __name__ == "__main__":
    main()