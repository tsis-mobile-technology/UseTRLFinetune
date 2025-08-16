#!/usr/bin/env python3
"""
Ollama GPT-OSS-20B Unsloth Fine-tuning Script
RTX 3060 (12GB VRAM) 최적화 버전
"""

import os
import sys
import json
import time
import argparse
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datasets import Dataset
import gc

# Unsloth imports
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class GPTOSSFineTuner:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 max_seq_length: int = 2048,
                 load_in_4bit: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
                 target_modules: Optional[list] = None):
        """
        GPT-OSS-20B Fine-tuner 초기화
        
        Args:
            model_name: 모델 이름 (Unsloth 지원 모델)
            max_seq_length: 최대 시퀀스 길이
            load_in_4bit: 4-bit 양자화 사용 여부
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: LoRA 타겟 모듈들
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # GPT-2/DialoGPT용 target modules
        self.target_modules = target_modules or [
            "c_attn", "c_proj", "c_fc"
        ]
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 모델과 토크나이저 초기화
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # CUDA 메모리 최적화
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def load_model(self):
        """모델과 토크나이저 로드"""
        self.logger.info(f"모델 로딩 시작: {self.model_name}")
        
        try:
            # Unsloth FastLanguageModel로 모델 로드
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto detection
                load_in_4bit=self.load_in_4bit,
                trust_remote_code=True,
            )
            
            self.logger.info("모델 로드 완료")
            
            # LoRA 어댑터 적용
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_r,
                target_modules=self.target_modules,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",  # Unsloth 최적화
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            
            self.logger.info("LoRA 어댑터 적용 완료")
            
            # 메모리 상태 출력
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU 메모리 사용량: {memory_used:.1f}GB / {memory_total:.1f}GB")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise
    
    def load_dataset(self, data_path: str) -> Dataset:
        """JSONL 데이터셋 로드 및 전처리"""
        self.logger.info(f"데이터셋 로딩: {data_path}")
        
        try:
            # JSONL 파일 읽기
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            self.logger.info(f"총 {len(data)}개 샘플 로드")
            
            # 데이터 형식 변환 (더 간단하고 안전하게)
            texts = []
            for item in data:
                # 텍스트 길이 제한 및 정제
                text = item['text'][:500]  # 500자 제한
                if len(text.strip()) > 10:  # 최소 길이를 10자로 완화
                    texts.append(text.strip())
            
            self.logger.info(f"유효한 텍스트 {len(texts)}개 선별")
            
            # Dataset 객체 생성
            dataset = Dataset.from_dict({"text": texts})
            
            # 토크나이징 (더 안전한 방식)
            def tokenize_function(examples):
                # 패딩 토큰 확인
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=512,  # 더 짧은 길이로 설정
                    padding="max_length",
                    return_tensors=None,
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
            
            self.logger.info(f"토크나이징 완료: {len(tokenized_dataset)} 샘플")
            return tokenized_dataset
            
        except Exception as e:
            self.logger.error(f"데이터셋 로드 실패: {e}")
            raise
    
    def setup_training_args(self, 
                          output_dir: str = "./results",
                          num_train_epochs: int = 1,
                          per_device_train_batch_size: int = 2,
                          gradient_accumulation_steps: int = 4,
                          learning_rate: float = 2e-4,
                          warmup_steps: int = 5,
                          logging_steps: int = 1,
                          save_steps: int = 10) -> TrainingArguments:
        """RTX 3060 최적화 훈련 인자 설정"""
        
        training_args = TrainingArguments(
            # 기본 설정
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # 학습률 및 옵티마이저
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            optim="adamw_8bit",  # 8-bit AdamW (메모리 절약)
            
            # 정밀도 설정 (RTX 3060은 bf16 지원)
            fp16=False,
            bf16=True,
            
            # 메모리 최적화
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            group_by_length=True,
            
            # 로깅 및 저장
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=False,
            
            # 기타
            report_to=None,  # wandb 비활성화 (필요시 "wandb"로 변경)
            run_name=f"gpt-oss-korean-{int(time.time())}",
        )
        
        self.logger.info("훈련 인자 설정 완료")
        self.logger.info(f"배치 크기: {per_device_train_batch_size}")
        self.logger.info(f"그래디언트 누적: {gradient_accumulation_steps}")
        self.logger.info(f"유효 배치 크기: {per_device_train_batch_size * gradient_accumulation_steps}")
        
        return training_args
    
    def train(self, 
              dataset: Dataset,
              training_args: TrainingArguments):
        """모델 훈련 실행"""
        self.logger.info("훈련 시작")
        
        try:
            # SFTTrainer 설정 (더 안전한 설정)
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=512,  # 더 짧은 길이
                args=training_args,
                packing=False,  # 메모리 최적화를 위해 비활성화
                formatting_func=None,  # 기본 포맷팅 사용
            )
            
            # 훈련 시작 전 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()
            
            self.logger.info("훈련 실행 중...")
            start_time = time.time()
            
            # 훈련 실행
            train_result = self.trainer.train()
            
            end_time = time.time()
            training_time = end_time - start_time
            
            self.logger.info(f"훈련 완료! 소요 시간: {training_time:.2f}초")
            self.logger.info(f"최종 Loss: {train_result.training_loss:.4f}")
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"훈련 실패: {e}")
            raise
    
    def save_model(self, output_path: str = "my_korean_gpt_oss_20b_lora"):
        """LoRA 어댑터 저장"""
        self.logger.info(f"모델 저장: {output_path}")
        
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_path, exist_ok=True)
            
            # LoRA 어댑터 저장
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            # 훈련 설정 저장
            config = {
                "model_name": self.model_name,
                "max_seq_length": self.max_seq_length,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.target_modules,
                "training_completed": True
            }
            
            with open(f"{output_path}/training_config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info("모델 저장 완료")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def test_inference(self, prompt: str = "안녕하세요! "):
        """간단한 추론 테스트"""
        self.logger.info("추론 테스트 시작")
        
        try:
            # FastLanguageModel을 추론 모드로 전환
            FastLanguageModel.for_inference(self.model)
            
            # 입력 토크나이징
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # 추론 실행
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 결과 디코딩
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            self.logger.info(f"입력: {prompt}")
            self.logger.info(f"출력: {generated_text}")
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"추론 테스트 실패: {e}")
            return None

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Ollama GPT-OSS-20B Korean Fine-tuning")
    parser.add_argument("--data-path", type=str, default="data/korean_wikipedia_data.jsonl",
                       help="훈련 데이터 경로")
    parser.add_argument("--model-name", type=str, default="microsoft/DialoGPT-medium",
                       help="베이스 모델 이름")
    parser.add_argument("--output-dir", type=str, default="my_korean_gpt_oss_20b_lora",
                       help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=1,
                       help="훈련 에포크 수")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="학습률")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                       help="최대 시퀀스 길이")
    parser.add_argument("--test-only", action="store_true",
                       help="테스트만 실행 (훈련 스킵)")
    
    args = parser.parse_args()
    
    # GPU 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다. GPU가 필요합니다.")
        sys.exit(1)
    
    print("🚀 Ollama GPT-OSS-20B Korean Fine-tuning 시작")
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # Fine-tuner 초기화
        fine_tuner = GPTOSSFineTuner(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length
        )
        
        # 모델 로드
        fine_tuner.load_model()
        
        if not args.test_only:
            # 데이터셋 로드
            if not Path(args.data_path).exists():
                print(f"❌ 데이터 파일을 찾을 수 없습니다: {args.data_path}")
                print("💡 먼저 다음 명령어로 데이터를 수집하세요:")
                print("   python scrape_wiki.py --max-articles 50")
                sys.exit(1)
            
            dataset = fine_tuner.load_dataset(args.data_path)
            
            # 훈련 인자 설정
            training_args = fine_tuner.setup_training_args(
                output_dir=f"./trainer_output/{args.output_dir}",
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
            # 훈련 실행
            fine_tuner.train(dataset, training_args)
            
            # 모델 저장
            fine_tuner.save_model(args.output_dir)
        
        # 추론 테스트
        print("\n🧪 추론 테스트:")
        test_prompts = [
            "안녕하세요! ",
            "대한민국의 수도는 ",
            "인공지능이란 ",
            "위키백과는 "
        ]
        
        for prompt in test_prompts:
            result = fine_tuner.test_inference(prompt)
            if result:
                print(f"✅ {prompt} → {result}")
            else:
                print(f"❌ {prompt} → 추론 실패")
        
        print("\n🎉 Fine-tuning 완료!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()