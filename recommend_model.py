#!/usr/bin/env python3
"""
RTX 3060에 최적화된 모델 추천 및 자동 실행 스크립트
"""

import torch
import subprocess
import sys

def recommend_model():
    """RTX 3060에 최적화된 모델 추천"""
    print("🔍 RTX 3060에 최적화된 모델 추천")
    print("=" * 50)
    
    # GPU 메모리 확인
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"📊 사용 가능한 VRAM: {total_memory:.1f}GB")
    
    models = [
        {
            "name": "microsoft/DialoGPT-medium",
            "size": "345M",
            "memory": "~2GB",
            "speed": "⚡ 매우 빠름",
            "stability": "🛡️ 매우 안정",
            "description": "대화형 모델, 한국어 fine-tuning에 최적화"
        },
        {
            "name": "microsoft/DialoGPT-large", 
            "size": "774M",
            "memory": "~4GB",
            "speed": "⚡ 빠름",
            "stability": "🛡️ 안정",
            "description": "더 큰 성능, 여전히 RTX 3060에서 안전"
        },
        {
            "name": "openai/gpt-oss-20b",
            "size": "20B",
            "memory": "~10-12GB",
            "speed": "🐌 느림",
            "stability": "⚠️ 불안정",
            "description": "최고 성능이지만 메모리 부족 위험"
        }
    ]
    
    print("\n📋 추천 모델 목록:")
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   크기: {model['size']}")
        print(f"   메모리: {model['memory']}")
        print(f"   속도: {model['speed']}")
        print(f"   안정성: {model['stability']}")
        print(f"   설명: {model['description']}")
    
    # 추천
    if total_memory < 8.0:
        recommended = 1  # DialoGPT-medium
        print(f"\n💡 {total_memory:.1f}GB VRAM에는 모델 1번을 강력히 추천합니다!")
    elif total_memory < 10.0:
        recommended = 2  # DialoGPT-large
        print(f"\n💡 {total_memory:.1f}GB VRAM에는 모델 2번을 추천합니다!")
    else:
        recommended = 3  # gpt-oss-20b
        print(f"\n💡 {total_memory:.1f}GB VRAM에는 모델 3번도 시도해볼 수 있습니다!")
    
    return models, recommended

def run_training(model_name: str):
    """선택된 모델로 훈련 실행"""
    print(f"\n🚀 {model_name} 모델로 훈련 시작...")
    
    # 모델별 최적화 매개변수
    if "medium" in model_name:
        cmd = [
            "python", "ollama_gpt-oss-20b_unsloth.py",
            "--model-name", model_name,
            "--batch-size", "2",
            "--epochs", "1",
            "--max-seq-length", "1024"
        ]
    elif "large" in model_name:
        cmd = [
            "python", "ollama_gpt-oss-20b_unsloth.py", 
            "--model-name", model_name,
            "--batch-size", "1",
            "--epochs", "1",
            "--max-seq-length", "1024"
        ]
    else:  # 20B 모델
        cmd = [
            "python", "ollama_gpt-oss-20b_unsloth.py",
            "--model-name", model_name,
            "--batch-size", "1", 
            "--epochs", "1",
            "--max-seq-length", "512"
        ]
    
    print(f"📝 실행 명령어: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n🎉 {model_name} 훈련 완료!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 훈련 실패: {e}")
        print("\n💡 다른 모델을 시도해보세요.")

def main():
    """메인 함수"""
    print("🤖 RTX 3060 최적화 모델 추천 도구")
    
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다.")
        return
    
    models, recommended = recommend_model()
    
    print(f"\n🎯 추천 모델: {recommended}번")
    
    while True:
        try:
            choice = input(f"\n선택하세요 (1-3, 기본값: {recommended}): ").strip()
            
            if not choice:
                choice = recommended
            else:
                choice = int(choice)
            
            if 1 <= choice <= 3:
                selected_model = models[choice-1]["name"]
                print(f"\n✅ 선택된 모델: {selected_model}")
                
                # 확인
                confirm = input("훈련을 시작하시겠습니까? (Y/n): ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    run_training(selected_model)
                    break
                else:
                    print("취소되었습니다.")
                    break
            else:
                print("❌ 1-3 사이의 숫자를 입력하세요.")
                
        except ValueError:
            print("❌ 올바른 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
            break

if __name__ == "__main__":
    main()