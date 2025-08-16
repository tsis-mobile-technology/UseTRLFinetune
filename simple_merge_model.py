#!/usr/bin/env python3
"""
간단한 LoRA 어댑터 병합 스크립트
transformers 라이브러리만 사용하여 Unsloth 의존성 회피
"""

import os
import torch
import json
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simple_merge.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def simple_merge_model(
    lora_path: str = "my_korean_gpt_oss_20b_lora",
    output_path: str = "my_korean_gpt_oss_20b_merged",
    base_model: str = "microsoft/DialoGPT-medium"
):
    """
    transformers와 PEFT를 사용한 간단한 모델 병합
    """
    logger = setup_logging()
    
    try:
        logger.info("🚀 간단한 모델 병합 프로세스 시작")
        
        # GPU 상태 확인
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            logger.warning("CUDA를 사용할 수 없습니다. CPU로 진행합니다.")
        
        # LoRA 어댑터 경로 확인
        lora_path = Path(lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA 어댑터를 찾을 수 없습니다: {lora_path}")
        
        logger.info(f"📁 LoRA 어댑터 경로: {lora_path}")
        
        # PEFT 설정 로드
        peft_config = PeftConfig.from_pretrained(str(lora_path))
        logger.info(f"📋 기본 모델: {peft_config.base_model_name_or_path}")
        
        # 기본 모델 로드
        logger.info("📥 기본 모델 로드 중...")
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,  # 메모리 절약
            low_cpu_mem_usage=True
        )
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
        
        logger.info("✅ 기본 모델 로드 완료")
        
        # LoRA 어댑터를 기본 모델에 로드
        logger.info("🔄 LoRA 어댑터 병합 중...")
        model_with_lora = PeftModel.from_pretrained(base_model, str(lora_path))
        
        # 병합 실행
        merged_model = model_with_lora.merge_and_unload()
        
        logger.info("✅ 모델 병합 완료")
        
        # 출력 디렉토리 생성
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # 병합된 모델 저장
        logger.info(f"💾 병합된 모델 저장 중: {output_path}")
        merged_model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        # 모델 정보 저장
        model_info = {
            "base_model": peft_config.base_model_name_or_path,
            "lora_adapter": str(lora_path),
            "merged_at": "2025-08-16",
            "model_type": "merged_lora",
            "korean_finetuned": True,
            "peft_config": {
                "r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "target_modules": list(peft_config.target_modules) if isinstance(peft_config.target_modules, set) else peft_config.target_modules,
                "lora_dropout": peft_config.lora_dropout
            }
        }
        
        with open(output_path / "model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        # 파일 크기 확인
        model_files = list(output_path.glob("*.safetensors")) + list(output_path.glob("*.bin"))
        if model_files:
            total_size = sum(f.stat().st_size for f in model_files) / 1024**3
            logger.info(f"📊 총 모델 크기: {total_size:.1f}GB")
        
        logger.info(f"✅ 모델 저장 완료!")
        logger.info(f"📁 저장 위치: {output_path}")
        
        # 간단한 추론 테스트
        logger.info("🧪 병합된 모델 추론 테스트")
        test_simple_inference(merged_model, tokenizer)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 병합 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_inference(model, tokenizer, prompts=None):
    """병합된 모델 추론 테스트"""
    logger = logging.getLogger(__name__)
    
    if prompts is None:
        prompts = [
            "안녕하세요! ",
            "대한민국의 수도는 ",
            "한글은 "
        ]
    
    try:
        # 모델을 평가 모드로 전환
        model.eval()
        
        for prompt in prompts:
            # 입력 토크나이징
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(model.device)
            
            # 추론 실행
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 결과 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ {prompt} → {generated_text}")
            
    except Exception as e:
        logger.error(f"❌ 추론 테스트 실패: {e}")

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="간단한 LoRA 모델 병합 도구")
    parser.add_argument("--lora-path", type=str, default="my_korean_gpt_oss_20b_lora",
                       help="LoRA 어댑터 경로")
    parser.add_argument("--output-path", type=str, default="my_korean_gpt_oss_20b_merged",
                       help="병합된 모델 저장 경로")
    
    args = parser.parse_args()
    
    print("🔄 간단한 LoRA 모델 병합 도구")
    print(f"📁 LoRA 경로: {args.lora_path}")
    print(f"💾 출력 경로: {args.output_path}")
    print()
    
    success = simple_merge_model(
        lora_path=args.lora_path,
        output_path=args.output_path
    )
    
    if success:
        print("\n🎉 모델 병합 성공!")
        print(f"📁 병합된 모델: {args.output_path}")
        print("💡 다음 단계: GGUF 변환으로 Ollama에서 사용 가능")
    else:
        print("\n❌ 모델 병합 실패")
        exit(1)

if __name__ == "__main__":
    main()