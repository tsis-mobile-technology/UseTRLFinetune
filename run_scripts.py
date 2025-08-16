#!/usr/bin/env python3
"""
통합 실행 스크립트 모음
기존의 run_로 시작하는 모든 스크립트들을 하나로 통합
"""

import os
import sys
import subprocess
import argparse
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
PROJECT_ROOT = "/media/proidea/hdd/Programming/UseTRLFinetune"
PYTHON_ENV = "/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python"

def run_command(cmd, timeout=900, capture_output=True):
    """명령어 실행 헬퍼 함수"""
    try:
        logger.info(f"실행 중: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        result = subprocess.run(
            cmd, 
            timeout=timeout, 
            capture_output=capture_output, 
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, 'PYTHONPATH': f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"}
        )
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"명령어 실행 시간 초과 ({timeout}초)")
        return None
    except Exception as e:
        logger.error(f"명령어 실행 실패: {e}")
        return None

def cpu_train():
    """CPU 모드 훈련 실행 (기존 run_cpu_train.sh)"""
    logger.info("=== CPU 모드 훈련 시작 ===")
    
    cmd = [
        PYTHON_ENV, "continue_train.py",
        "--base-model", "EleutherAI/polyglot-ko-1.3b",
        "--finetuned-model", "my_optimized_model",
        "--data-path", "web_data_add1.jsonl",
        "--output-dir", "my_continued_model_cpu",
        "--mode", "continue",
        "--batch-size", "1",
        "--lora-r", "32",
        "--lora-alpha", "64",
        "--max-length", "512",
        "--epochs", "5"
    ]
    
    result = run_command(cmd, timeout=3600, capture_output=False)  # 1시간 제한, 실시간 출력
    
    if result and result.returncode == 0:
        logger.info("✅ CPU 모드 훈련 완료")
        return True
    else:
        logger.error("❌ CPU 모드 훈련 실패")
        return False

def test_create_reference_model():
    """참조 모델 생성 테스트 (기존 run_create_ref_test.py)"""
    logger.info("=== 참조 모델 생성 테스트 시작 ===")
    
    cmd = [
        PYTHON_ENV, "train.py",
        "--model_name", "EleutherAI/polyglot-ko-1.3b",
        "--dataset_name", "maywell/korean_textbooks",
        "--output_dir", "test_create_ref_model",
        "--batch_size", "1",
        "--mini_batch_size", "1",
        "--max_ppo_steps", "1",
        "--dataset_sample_size", "3",
        "--max_new_tokens", "8",
        "--learning_rate", "1e-5"
    ]
    
    result = run_command(cmd, timeout=900)
    
    if result:
        logger.info("=== STDOUT ===")
        logger.info(result.stdout)
        logger.info("=== STDERR ===") 
        logger.info(result.stderr)
        logger.info(f"=== Return code: {result.returncode} ===")
        
        # 특정 오류 패턴 확인
        if "'tuple' object has no attribute 'logits'" in result.stderr:
            logger.error("❌ tuple/logits AttributeError 여전히 발생!")
            return False
        elif "AttributeError" in result.stderr and "logits" in result.stderr:
            logger.warning("⚠️ 다른 logits 관련 AttributeError 감지")
            return False
        elif result.returncode == 0:
            logger.info("✅ 스크립트 성공적으로 완료!")
            return True
        else:
            # 진행 상황 체크
            if "참조 모델 생성 완료" in result.stdout or "✅ 참조 모델 생성 완료" in result.stdout:
                logger.info("✅ 참조 모델 생성 단계 통과!")
            if "✅ PPOTrainer 생성 성공" in result.stdout:
                logger.info("✅ PPOTrainer 생성 단계 통과!")
            return False
    else:
        logger.error("❌ 명령어 실행 실패")
        return False

def debug_reference_model():
    """참조 모델 디버깅 실행 (기존 run_debug_ref.sh)"""
    logger.info("=== 참조 모델 디버깅 시작 ===")
    
    cmd = [PYTHON_ENV, "debug_ref_model_output.py"]
    result = run_command(cmd, capture_output=False)
    
    if result and result.returncode == 0:
        logger.info("✅ 참조 모델 디버깅 완료")
        return True
    else:
        logger.error("❌ 참조 모델 디버깅 실패")
        return False

def show_fixed_train_guide():
    """수정된 훈련 스크립트 가이드 출력 (기존 run_fixed_train.sh)"""
    logger.info("=== 수정된 PPO 훈련 가이드 ===")
    
    # GPU 메모리 상태 확인
    logger.info("📊 현재 GPU 메모리 상태:")
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if gpu_result.returncode == 0:
            logger.info(gpu_result.stdout.strip())
        else:
            logger.warning("GPU 메모리 정보를 가져올 수 없습니다.")
    except:
        logger.warning("nvidia-smi를 실행할 수 없습니다.")
    
    # 가이드 출력
    guides = [
        {
            "name": "🚀 테스트 1: 8-bit 양자화 + 메모리 최적화 모드",
            "command": [
                "python train_fixed.py \\",
                "  --model_name EleutherAI/polyglot-ko-1.3b \\",
                "  --dataset_name maywell/korean_textbooks \\",
                "  --output_dir my_finetune_model_fixed_250627 \\",
                "  --batch_size 2 \\",
                "  --mini_batch_size 1 \\",
                "  --input_max_text_length 256 \\",
                "  --max_new_tokens 16 \\",
                "  --dataset_sample_size 20 \\",
                "  --max_ppo_steps 3 \\",
                "  --lora_r 16 \\",
                "  --lora_alpha 32 \\",
                "  --use_8bit_quantization \\",
                "  --enable_gradient_checkpointing \\",
                "  --max_memory_per_gpu 10GB"
            ]
        },
        {
            "name": "🔧 테스트 2: 최소 메모리 모드 (극한 절약)",
            "command": [
                "python train_fixed.py \\",
                "  --model_name EleutherAI/polyglot-ko-1.3b \\",
                "  --dataset_name maywell/korean_textbooks \\",
                "  --output_dir my_finetune_model_minimal_250627 \\",
                "  --batch_size 1 \\",
                "  --mini_batch_size 1 \\",
                "  --input_max_text_length 128 \\",
                "  --max_new_tokens 8 \\",
                "  --dataset_sample_size 10 \\",
                "  --max_ppo_steps 2 \\",
                "  --lora_r 8 \\",
                "  --lora_alpha 16 \\",
                "  --use_8bit_quantization \\",
                "  --enable_gradient_checkpointing \\",
                "  --max_memory_per_gpu 8GB"
            ]
        },
        {
            "name": "⚡ 테스트 3: 양자화 없는 모드 (더 나은 성능)",
            "command": [
                "python train_fixed.py \\",
                "  --model_name EleutherAI/polyglot-ko-1.3b \\",
                "  --dataset_name maywell/korean_textbooks \\",
                "  --output_dir my_finetune_model_fp16_250627 \\",
                "  --batch_size 2 \\",
                "  --mini_batch_size 1 \\",
                "  --input_max_text_length 256 \\",
                "  --max_new_tokens 16 \\",
                "  --dataset_sample_size 20 \\",
                "  --max_ppo_steps 3 \\",
                "  --lora_r 16 \\",
                "  --lora_alpha 32 \\",
                "  --enable_gradient_checkpointing \\",
                "  --max_memory_per_gpu 10GB"
            ]
        }
    ]
    
    for guide in guides:
        logger.info(f"\n{guide['name']}")
        logger.info("실행 명령어:")
        for line in guide["command"]:
            logger.info(line)
    
    logger.info("\n📝 주요 개선사항:")
    improvements = [
        "✅ 'tuple' object has no attribute 'logits' 오류 완전 해결",
        "✅ CUDA Out of Memory 오류 방지를 위한 메모리 최적화",
        "✅ 8-bit 양자화 지원으로 메모리 사용량 50% 감소",
        "✅ Gradient checkpointing으로 추가 메모리 절약",
        "✅ 동적 메모리 관리 및 오류 복구",
        "✅ 한국어 데이터셋 완전 지원"
    ]
    
    for improvement in improvements:
        logger.info(improvement)
    
    logger.info("\n🎯 권장 시작 명령어: 테스트 1 (8-bit 양자화 모드)")

def test_reference_model_fix():
    """참조 모델 수정 테스트 (기존 run_test_ref_fix.py)"""
    logger.info("=== 참조 모델 수정 테스트 시작 ===")
    
    cmd = [
        PYTHON_ENV, "train.py",
        "--model_name", "EleutherAI/polyglot-ko-1.3b",
        "--dataset_name", "maywell/korean_textbooks",
        "--output_dir", "my_ref_fixed_test",
        "--batch_size", "1",
        "--mini_batch_size", "1",
        "--max_ppo_steps", "1",
        "--dataset_sample_size", "5",
        "--max_new_tokens", "8",
        "--learning_rate", "1e-5"
    ]
    
    result = run_command(cmd, timeout=600)
    
    if result:
        logger.info("STDOUT:")
        logger.info(result.stdout)
        logger.info("\nSTDERR:")
        logger.info(result.stderr)
        logger.info(f"\nReturn code: {result.returncode}")
        
        # 특정 오류 패턴 확인
        if "'tuple' object has no attribute 'logits'" in result.stderr:
            logger.error("❌ tuple/logits AttributeError 여전히 발생!")
            return False
        elif "AttributeError" in result.stderr:
            logger.warning("⚠️ 다른 AttributeError 감지")
            return False
        elif result.returncode == 0:
            logger.info("✅ 스크립트 성공적으로 완료!")
            return True
        else:
            logger.warning(f"⚠️ 스크립트가 return code {result.returncode}로 실패")
            return False
    else:
        logger.error("❌ 명령어 실행 실패")
        return False

def train_test():
    """일반 훈련 테스트 (기존 run_train_test.sh)"""
    logger.info("=== 일반 훈련 테스트 시작 ===")
    
    cmd = [
        PYTHON_ENV, "train.py",
        "--model_name", "EleutherAI/polyglot-ko-1.3b",
        "--dataset_name", "maywell/korean_textbooks",
        "--output_dir", "my_ref_model_test",
        "--batch_size", "2",
        "--mini_batch_size", "1",
        "--max_ppo_steps", "1",
        "--dataset_sample_size", "10",
        "--max_new_tokens", "16",
        "--learning_rate", "1e-5"
    ]
    
    result = run_command(cmd, capture_output=False)
    
    if result and result.returncode == 0:
        logger.info("✅ 일반 훈련 테스트 완료")
        return True
    else:
        logger.error("❌ 일반 훈련 테스트 실패")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 실행 스크립트 모음")
    parser.add_argument(
        "command",
        choices=[
            "cpu-train", "test-ref-create", "debug-ref", "guide-fixed",
            "test-ref-fix", "train-test", "list"
        ],
        help="실행할 명령"
    )
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("\n=== 사용 가능한 명령어 ===")
        commands = {
            "cpu-train": "CPU 모드로 훈련 실행",
            "test-ref-create": "참조 모델 생성 테스트",
            "debug-ref": "참조 모델 디버깅 실행",
            "guide-fixed": "수정된 훈련 스크립트 가이드 출력",
            "test-ref-fix": "참조 모델 수정 테스트",
            "train-test": "일반 훈련 테스트",
            "list": "이 도움말 출력"
        }
        
        for cmd, desc in commands.items():
            print(f"  {cmd:15} : {desc}")
        
        print(f"\n사용법: python {sys.argv[0]} <command>")
        return
    
    # 작업 디렉토리 확인 및 변경
    if not os.path.exists(PROJECT_ROOT):
        logger.error(f"프로젝트 루트 디렉토리를 찾을 수 없습니다: {PROJECT_ROOT}")
        return
    
    os.chdir(PROJECT_ROOT)
    
    # 명령어 실행
    command_map = {
        "cpu-train": cpu_train,
        "test-ref-create": test_create_reference_model,
        "debug-ref": debug_reference_model,
        "guide-fixed": show_fixed_train_guide,
        "test-ref-fix": test_reference_model_fix,
        "train-test": train_test
    }
    
    success = command_map[args.command]()
    
    if success:
        logger.info(f"✅ '{args.command}' 명령이 성공적으로 완료되었습니다.")
    else:
        logger.error(f"❌ '{args.command}' 명령이 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()