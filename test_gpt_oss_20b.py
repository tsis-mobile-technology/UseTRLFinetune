#!/usr/bin/env python3
"""
GPT-OSS-20B 모델 테스트 스크립트
RTX 3060에서 메모리 사용량 확인 및 기본 동작 테스트
"""

import torch
import sys
from pathlib import Path

def check_model_compatibility():
    """모델 호환성 및 메모리 요구사항 체크"""
    print("🔍 GPT-OSS-20B 모델 호환성 체크")
    
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return False
    
    # GPU 정보
    gpu_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎮 GPU: {gpu_name}")
    print(f"📊 총 VRAM: {total_memory:.1f}GB")
    
    # 메모리 요구사항 체크
    if total_memory < 10.0:
        print(f"⚠️  경고: {total_memory:.1f}GB는 20B 모델에 부족할 수 있습니다.")
        print("💡 권장사항:")
        print("   1. 더 작은 모델 사용: microsoft/DialoGPT-medium")
        print("   2. 또는 매우 작은 배치 사이즈 사용: --batch-size 1")
        return False
    else:
        print(f"✅ {total_memory:.1f}GB는 20B 모델 실행에 충분합니다.")
        return True

def test_model_loading():
    """실제 모델 로딩 테스트"""
    print("\n🚀 모델 로딩 테스트 시작...")
    
    try:
        # Unsloth import 체크
        from unsloth import FastLanguageModel
        print("✅ Unsloth 라이브러리 정상")
        
        # 메모리 정리
        torch.cuda.empty_cache()
        
        print("📥 GPT-OSS-20B 모델 로딩 시도...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="openai/gpt-oss-20b",
            max_seq_length=1024,  # 작은 길이로 테스트
            dtype=None,
            load_in_4bit=True,  # 4-bit 양자화 필수
            trust_remote_code=True,
        )
        
        # 메모리 사용량 체크
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ 모델 로딩 성공!")
        print(f"📊 GPU 메모리 사용량: {memory_used:.1f}GB / {memory_total:.1f}GB")
        print(f"📈 메모리 사용률: {memory_used/memory_total*100:.1f}%")
        
        # 간단한 토큰 테스트
        inputs = tokenizer("안녕하세요!", return_tensors="pt")
        print(f"✅ 토크나이저 정상 동작")
        
        return True
        
    except ImportError as e:
        print(f"❌ 라이브러리 오류: {e}")
        print("💡 다음 명령어로 설치하세요:")
        print('   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        return False
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        print("💡 가능한 해결책:")
        print("   1. 더 작은 모델 사용: microsoft/DialoGPT-medium")
        print("   2. 메모리 정리 후 재시도")
        print("   3. 시스템 재부팅 후 재시도")
        return False

def main():
    """메인 함수"""
    print("🔬 GPT-OSS-20B 모델 테스트 도구")
    print("=" * 50)
    
    # 1. 호환성 체크
    if not check_model_compatibility():
        print("\n💡 더 안전한 옵션으로 진행하려면:")
        print("   python ollama_gpt-oss-20b_unsloth.py --model-name microsoft/DialoGPT-medium")
        response = input("\n그래도 20B 모델을 테스트하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("테스트 중단")
            return
    
    # 2. 실제 모델 로딩 테스트
    success = test_model_loading()
    
    if success:
        print("\n🎉 테스트 성공! GPT-OSS-20B 모델을 사용할 수 있습니다.")
        print("🚀 Fine-tuning을 시작하려면:")
        print("   python ollama_gpt-oss-20b_unsloth.py --batch-size 1 --epochs 1")
    else:
        print("\n🔄 대안 방법:")
        print("   python ollama_gpt-oss-20b_unsloth.py --model-name microsoft/DialoGPT-medium")

if __name__ == "__main__":
    main()