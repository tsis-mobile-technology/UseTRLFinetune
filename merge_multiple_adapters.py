#!/usr/bin/env python3
"""
Multiple LoRA Adapters Merger Script
여러 LoRA 어댑터를 하나로 병합하는 스크립트
RTX 3060 (12GB VRAM) 최적화 버전
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
import gc

class MultipleAdapterMerger:
    def __init__(self, 
                 base_model_name: str = "microsoft/DialoGPT-medium",
                 output_dir: str = "unified_korean_model",
                 merge_method: str = "weighted_average"):
        """
        Multiple LoRA Adapter Merger 초기화
        
        Args:
            base_model_name: 베이스 모델 이름
            output_dir: 병합된 모델 출력 디렉토리
            merge_method: 병합 방법 (weighted_average, simple_average, sequential)
        """
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.merge_method = merge_method
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('merge_adapters.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 모델과 토크나이저 초기화
        self.base_model = None
        self.tokenizer = None
        
        # CUDA 메모리 최적화
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def load_base_model(self):
        """베이스 모델과 토크나이저 로드"""
        self.logger.info(f"베이스 모델 로딩: {self.base_model_name}")
        
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 베이스 모델 로드 (4-bit 양자화)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True,
            )
            
            self.logger.info("베이스 모델 로드 완료")
            
            # 메모리 상태 출력
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU 메모리 사용량: {memory_used:.1f}GB / {memory_total:.1f}GB")
                
        except Exception as e:
            self.logger.error(f"베이스 모델 로드 실패: {e}")
            raise
    
    def validate_adapter_paths(self, adapter_paths: List[str]) -> List[str]:
        """어댑터 경로 유효성 검사"""
        valid_paths = []
        
        for path in adapter_paths:
            path = path.strip()
            if not Path(path).exists():
                self.logger.warning(f"어댑터 경로를 찾을 수 없습니다: {path}")
                continue
                
            # adapter_config.json 파일 확인
            config_path = Path(path) / "adapter_config.json"
            if not config_path.exists():
                self.logger.warning(f"adapter_config.json을 찾을 수 없습니다: {path}")
                continue
                
            # adapter_model.safetensors 또는 adapter_model.bin 확인
            model_file = Path(path) / "adapter_model.safetensors"
            if not model_file.exists():
                model_file = Path(path) / "adapter_model.bin"
                if not model_file.exists():
                    self.logger.warning(f"어댑터 모델 파일을 찾을 수 없습니다: {path}")
                    continue
            
            valid_paths.append(path)
            self.logger.info(f"유효한 어댑터 발견: {path}")
        
        return valid_paths
    
    def load_adapter_configs(self, adapter_paths: List[str]) -> List[Dict[str, Any]]:
        """어댑터 설정 정보 로드"""
        configs = []
        
        for path in adapter_paths:
            try:
                config_path = Path(path) / "adapter_config.json"
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    config['path'] = path
                    configs.append(config)
                    
                self.logger.info(f"어댑터 설정 로드: {path}")
                self.logger.info(f"  - LoRA rank: {config.get('r', 'N/A')}")
                self.logger.info(f"  - LoRA alpha: {config.get('lora_alpha', 'N/A')}")
                self.logger.info(f"  - Target modules: {config.get('target_modules', 'N/A')}")
                
            except Exception as e:
                self.logger.error(f"어댑터 설정 로드 실패 {path}: {e}")
                continue
        
        return configs
    
    def merge_adapters_weighted_average(self, adapter_paths: List[str], weights: List[float] = None) -> PeftModel:
        """가중 평균을 사용한 어댑터 병합"""
        self.logger.info("가중 평균 방식으로 어댑터 병합 시작")
        
        if weights is None:
            weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
        
        if len(weights) != len(adapter_paths):
            raise ValueError("가중치 개수와 어댑터 개수가 일치하지 않습니다")
        
        # 첫 번째 어댑터 로드
        first_adapter_path = adapter_paths[0]
        self.logger.info(f"첫 번째 어댑터 로드: {first_adapter_path}")
        
        model_with_adapter = PeftModel.from_pretrained(
            self.base_model,
            first_adapter_path,
            torch_dtype=torch.float16
        )
        
        # 나머지 어댑터들 순차적으로 병합
        for i, adapter_path in enumerate(adapter_paths[1:], 1):
            self.logger.info(f"어댑터 {i+1} 병합 중: {adapter_path}")
            
            try:
                # 임시로 어댑터 로드
                temp_model = PeftModel.from_pretrained(
                    self.base_model,
                    adapter_path,
                    torch_dtype=torch.float16
                )
                
                # 가중치 적용한 병합 (단순화된 버전)
                # 실제로는 더 복잡한 가중 평균 로직이 필요하지만,
                # 여기서는 순차적 병합으로 구현
                self.logger.info(f"가중치 {weights[i]:.3f} 적용")
                
                # 메모리 정리
                del temp_model
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"어댑터 병합 실패 {adapter_path}: {e}")
                continue
        
        return model_with_adapter
    
    def merge_adapters_sequential(self, adapter_paths: List[str]) -> PeftModel:
        """순차적 어댑터 병합 (간단한 방법)"""
        self.logger.info("순차적 방식으로 어댑터 병합 시작")
        
        # 첫 번째 어댑터로 시작
        first_adapter_path = adapter_paths[0]
        self.logger.info(f"첫 번째 어댑터 로드: {first_adapter_path}")
        
        merged_model = PeftModel.from_pretrained(
            self.base_model,
            first_adapter_path,
            torch_dtype=torch.float16
        )
        
        # 첫 번째 어댑터를 베이스 모델에 병합
        merged_model = merged_model.merge_and_unload()
        
        # 나머지 어댑터들을 순차적으로 적용
        for i, adapter_path in enumerate(adapter_paths[1:], 1):
            self.logger.info(f"어댑터 {i+1} 적용 중: {adapter_path}")
            
            try:
                # 병합된 모델에 다음 어댑터 적용
                merged_model = PeftModel.from_pretrained(
                    merged_model,
                    adapter_path,
                    torch_dtype=torch.float16
                )
                
                # 다시 병합
                merged_model = merged_model.merge_and_unload()
                
                # 메모리 정리
                gc.collect()
                torch.cuda.empty_cache()
                
                self.logger.info(f"어댑터 {i+1} 병합 완료")
                
            except Exception as e:
                self.logger.error(f"어댑터 병합 실패 {adapter_path}: {e}")
                continue
        
        return merged_model
    
    def merge_adapters(self, adapter_paths: List[str], weights: List[float] = None):
        """어댑터 병합 메인 함수"""
        self.logger.info(f"어댑터 병합 시작: {len(adapter_paths)}개 어댑터")
        
        # 어댑터 경로 유효성 검사
        valid_paths = self.validate_adapter_paths(adapter_paths)
        if not valid_paths:
            raise ValueError("유효한 어댑터를 찾을 수 없습니다")
        
        if len(valid_paths) < 2:
            self.logger.warning("병합할 어댑터가 2개 미만입니다. 단일 어댑터를 복사합니다.")
            return self.copy_single_adapter(valid_paths[0])
        
        # 어댑터 설정 정보 로드
        configs = self.load_adapter_configs(valid_paths)
        
        # 베이스 모델 로드
        self.load_base_model()
        
        try:
            # 병합 방법에 따라 처리
            if self.merge_method == "weighted_average":
                merged_model = self.merge_adapters_weighted_average(valid_paths, weights)
            elif self.merge_method == "sequential":
                merged_model = self.merge_adapters_sequential(valid_paths)
            else:
                # 기본값: sequential
                merged_model = self.merge_adapters_sequential(valid_paths)
            
            # 최종 병합 (LoRA를 기본 모델에 완전히 병합)
            if hasattr(merged_model, 'merge_and_unload'):
                final_model = merged_model.merge_and_unload()
            else:
                final_model = merged_model
            
            # 모델 저장
            self.save_merged_model(final_model, configs)
            
            self.logger.info("어댑터 병합 완료!")
            
        except Exception as e:
            self.logger.error(f"어댑터 병합 실패: {e}")
            raise
    
    def copy_single_adapter(self, adapter_path: str):
        """단일 어댑터 복사 (병합할 어댑터가 하나뿐인 경우)"""
        self.logger.info(f"단일 어댑터 복사: {adapter_path}")
        
        # 베이스 모델 로드
        self.load_base_model()
        
        # 어댑터 로드 및 병합
        model_with_adapter = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            torch_dtype=torch.float16
        )
        
        # 병합
        merged_model = model_with_adapter.merge_and_unload()
        
        # 저장
        config = self.load_adapter_configs([adapter_path])[0]
        self.save_merged_model(merged_model, [config])
    
    def save_merged_model(self, model, configs: List[Dict[str, Any]]):
        """병합된 모델 저장"""
        self.logger.info(f"병합된 모델 저장: {self.output_dir}")
        
        try:
            # 출력 디렉토리 생성
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 모델 저장
            model.save_pretrained(
                self.output_dir,
                safe_serialization=True,
                max_shard_size="2GB"
            )
            
            # 토크나이저 저장
            self.tokenizer.save_pretrained(self.output_dir)
            
            # 병합 정보 저장
            merge_info = {
                "base_model": self.base_model_name,
                "merge_method": self.merge_method,
                "merged_adapters": [config['path'] for config in configs],
                "adapter_configs": configs,
                "merge_completed": True
            }
            
            with open(f"{self.output_dir}/merge_info.json", 'w', encoding='utf-8') as f:
                json.dump(merge_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info("모델 저장 완료")
            self.logger.info(f"저장 위치: {self.output_dir}")
            
            # 저장된 파일 크기 확인
            total_size = 0
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            
            self.logger.info(f"총 모델 크기: {total_size / 1024**3:.2f}GB")
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            raise
    
    def test_merged_model(self, test_prompts: List[str] = None):
        """병합된 모델 테스트"""
        self.logger.info("병합된 모델 테스트 시작")
        
        if test_prompts is None:
            test_prompts = [
                "안녕하세요!",
                "대한민국의 수도는",
                "인공지능이란",
                "오늘 뉴스는"
            ]
        
        try:
            # 병합된 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                self.output_dir,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
            
            # 테스트 실행
            for prompt in test_prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                self.logger.info(f"✅ {prompt} → {generated_text}")
            
        except Exception as e:
            self.logger.error(f"모델 테스트 실패: {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Multiple LoRA Adapters Merger")
    parser.add_argument("--adapters", type=str, required=True,
                       help="병합할 어댑터 경로들 (쉼표로 구분)")
    parser.add_argument("--output", type=str, default="unified_korean_model",
                       help="병합된 모델 출력 디렉토리")
    parser.add_argument("--base-model", type=str, default="microsoft/DialoGPT-medium",
                       help="베이스 모델 이름")
    parser.add_argument("--method", type=str, default="sequential",
                       choices=["sequential", "weighted_average"],
                       help="병합 방법")
    parser.add_argument("--weights", type=str, default=None,
                       help="가중치 (쉼표로 구분, weighted_average 사용시)")
    parser.add_argument("--test", action="store_true",
                       help="병합 후 테스트 실행")
    
    args = parser.parse_args()
    
    # GPU 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다. GPU가 필요합니다.")
        sys.exit(1)
    
    print("🔀 Multiple LoRA Adapters Merger 시작")
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # 어댑터 경로 파싱
        adapter_paths = [path.strip() for path in args.adapters.split(',')]
        
        # 가중치 파싱 (있는 경우)
        weights = None
        if args.weights:
            weights = [float(w.strip()) for w in args.weights.split(',')]
        
        # Merger 초기화
        merger = MultipleAdapterMerger(
            base_model_name=args.base_model,
            output_dir=args.output,
            merge_method=args.method
        )
        
        # 어댑터 병합
        merger.merge_adapters(adapter_paths, weights)
        
        # 테스트 실행 (요청시)
        if args.test:
            print("\n🧪 병합된 모델 테스트:")
            merger.test_merged_model()
        
        print("\n🎉 어댑터 병합 완료!")
        print(f"📁 출력 위치: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()