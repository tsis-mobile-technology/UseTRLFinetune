#!/usr/bin/env python3
"""
수정된 train.py 스크립트를 테스트하는 스크립트
메모리 사용량을 더욱 줄여서 테스트합니다.
"""

import subprocess
import sys

def test_fixed_train():
    """수정된 train.py를 테스트"""
    cmd = [
        sys.executable, "train.py",
        "--model_name", "EleutherAI/polyglot-ko-1.3b",
        "--dataset_name", "maywell/korean_textbooks", 
        "--output_dir", "test_fixed_model",
        "--batch_size", "2",  # 더 작은 배치 크기
        "--mini_batch_size", "1",
        "--lora_r", "64",  # 더 작은 LoRA rank
        "--lora_alpha", "128",
        "--input_max_text_length", "1024",  # 더 짧은 시퀀스
        "--max_ppo_steps", "3",  # 매우 적은 스텝으로 테스트
        "--dataset_sample_size", "20",  # 더 작은 데이터셋
        "--max_new_tokens", "8",  # 더 적은 생성 토큰
        "--use_8bit_quantization",  # 메모리 절약
        "--max_memory_per_gpu", "8GB"
    ]
    
    print("수정된 train.py 실행 중...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 테스트 성공!")
        print("STDOUT:", result.stdout[-1000:])  # 마지막 1000자만 출력
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 테스트 실패!")
        print("STDERR:", e.stderr[-1000:])  # 마지막 1000자만 출력
        print("STDOUT:", e.stdout[-1000:])
        return False

if __name__ == "__main__":
    success = test_fixed_train()
    sys.exit(0 if success else 1)