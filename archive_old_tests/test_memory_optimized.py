#!/usr/bin/env python3
"""
메모리 최적화된 PPO 훈련 테스트 스크립트
"""

import torch
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_memory():
    """GPU 메모리 상태 확인"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        free_memory = total_memory - cached_memory
        
        logger.info(f"GPU 메모리 상태:")
        logger.info(f"  총 메모리: {total_memory:.2f} GB")
        logger.info(f"  할당된 메모리: {allocated_memory:.2f} GB")
        logger.info(f"  캐시된 메모리: {cached_memory:.2f} GB")
        logger.info(f"  사용 가능 메모리: {free_memory:.2f} GB")
        
        return free_memory
    else:
        logger.warning("CUDA를 사용할 수 없습니다.")
        return 0

def test_memory_optimized_training():
    """메모리 최적화된 훈련 테스트"""
    logger.info("=== 메모리 최적화된 PPO 훈련 테스트 시작 ===")
    
    # 초기 메모리 상태 확인
    check_gpu_memory()
    
    # 메모리 최적화 설정으로 훈련 실행
    test_commands = [
        # 기본 메모리 최적화 설정 (양자화 없음)
        [
            "python", "train.py",
            "--model_name", "EleutherAI/polyglot-ko-1.3b",
            "--batch_size", "1",
            "--mini_batch_size", "1", 
            "--input_max_text_length", "128",
            "--max_new_tokens", "8",
            "--dataset_sample_size", "10",
            "--max_ppo_steps", "2",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--enable_gradient_checkpointing",
            "--max_memory_per_gpu", "8GB"
        ],
        # 8bit 양자화와 함께
        [
            "python", "train.py",
            "--model_name", "EleutherAI/polyglot-ko-1.3b",
            "--batch_size", "2",
            "--mini_batch_size", "1",
            "--input_max_text_length", "256",
            "--max_new_tokens", "16",
            "--dataset_sample_size", "20",
            "--max_ppo_steps", "3",
            "--lora_r", "16",
            "--lora_alpha", "32",
            "--use_8bit_quantization",
            "--enable_gradient_checkpointing",
            "--max_memory_per_gpu", "10GB"
        ]
    ]
    
    for i, cmd in enumerate(test_commands):
        logger.info(f"\n--- 테스트 {i+1}: {' '.join(cmd[2:])} ---")
        
        # 메모리 상태 확인
        free_memory = check_gpu_memory()
        
        if free_memory < 2.0:  # 2GB 미만이면 경고
            logger.warning(f"사용 가능한 메모리가 부족합니다: {free_memory:.2f} GB")
            logger.info("더 작은 설정으로 조정합니다...")
            
            # 설정 축소
            if "--batch_size" in cmd:
                idx = cmd.index("--batch_size") + 1
                cmd[idx] = "1"
            if "--input_max_text_length" in cmd:
                idx = cmd.index("--input_max_text_length") + 1
                cmd[idx] = "64"
            if "--max_new_tokens" in cmd:
                idx = cmd.index("--max_new_tokens") + 1
                cmd[idx] = "4"
        
        logger.info(f"실행할 명령어: {' '.join(cmd)}")
        
        # 실제 실행 대신 명령어만 출력 (사용자가 직접 실행)
        logger.info("위 명령어를 터미널에서 실행해보세요.")
        
        print(f"\n{'='*50}")
        print(f"테스트 {i+1} 명령어:")
        print(' '.join(cmd))
        print(f"{'='*50}\n")

if __name__ == "__main__":
    test_memory_optimized_training()