#!/usr/bin/env python3
"""
LoRA 어댑터를 기본 모델과 병합하여 전체 모델로 저장
추론 속도 향상을 위한 모델 병합 스크립트
"""

import os
import torch
import argparse
from pathlib import Path
from unsloth import FastLanguageModel
import logging

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_merge.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def merge_and_save_model(
    lora_path: str = "my_korean_gpt_oss_20b_lora",
    output_path: str = "my_korean_gpt_oss_20b_merged",
    base_model: str = "microsoft/DialoGPT-medium"
):
    """
    LoRA 어댑터를 기본 모델과 병합하여 저장
    
    Args:
        lora_path: LoRA 어댑터 경로
        output_path: 병합된 모델 저장 경로
        base_model: 기본 모델 이름
    """
    logger = setup_logging()
    
    try:
        logger.info("🚀 모델 병합 프로세스 시작")
        
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
        
        logger.info(f"📁 LoRA 어댑터 로딩: {lora_path}")
        
        # 기본 모델과 LoRA 어댑터 로드
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(lora_path),  # LoRA 어댑터 경로
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,  # 병합할 때는 4bit 비활성화
        )
        
        logger.info("✅ LoRA 어댑터 로드 완료")
        
        # 메모리 사용량 확인
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU 메모리 사용량: {memory_used:.1f}GB")
        
        # LoRA 가중치를 기본 모델과 병합
        logger.info("🔄 LoRA 가중치 병합 중...")
        merged_model = model.merge_and_unload()
        
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
            "base_model": base_model,
            "lora_adapter": str(lora_path),
            "merged_at": "2025-08-16",
            "model_type": "merged_lora",
            "korean_finetuned": True
        }
        
        import json
        with open(output_path / "model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        # 파일 크기 확인
        model_files = list(output_path.glob("*.safetensors")) + list(output_path.glob("*.bin"))
        total_size = sum(f.stat().st_size for f in model_files) / 1024**3
        
        logger.info(f"✅ 모델 저장 완료!")
        logger.info(f"📊 총 모델 크기: {total_size:.1f}GB")
        logger.info(f"📁 저장 위치: {output_path}")
        
        # 간단한 추론 테스트
        logger.info("🧪 병합된 모델 추론 테스트")
        test_inference(merged_model, tokenizer)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 병합 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference(model, tokenizer, prompts=None):
    """병합된 모델 추론 테스트"""
    logger = logging.getLogger(__name__)
    
    if prompts is None:
        prompts = [
            "안녕하세요! ",
            "대한민국의 수도는 ",
            "인공지능이란 ",
            "한글은 "
        ]
    
    try:
        # 추론 모드로 전환
        FastLanguageModel.for_inference(model)
        
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
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 결과 디코딩
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ {prompt} → {generated_text}")
            
    except Exception as e:
        logger.error(f"❌ 추론 테스트 실패: {e}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="LoRA 모델 병합 도구")
    parser.add_argument("--lora-path", type=str, default="my_korean_gpt_oss_20b_lora",
                       help="LoRA 어댑터 경로")
    parser.add_argument("--output-path", type=str, default="my_korean_gpt_oss_20b_merged",
                       help="병합된 모델 저장 경로")
    parser.add_argument("--base-model", type=str, default="microsoft/DialoGPT-medium",
                       help="기본 모델 이름")
    
    args = parser.parse_args()
    
    print("🔄 LoRA 모델 병합 도구")
    print(f"📁 LoRA 경로: {args.lora_path}")
    print(f"💾 출력 경로: {args.output_path}")
    print(f"🏗️ 기본 모델: {args.base_model}")
    print()
    
    success = merge_and_save_model(
        lora_path=args.lora_path,
        output_path=args.output_path,
        base_model=args.base_model
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