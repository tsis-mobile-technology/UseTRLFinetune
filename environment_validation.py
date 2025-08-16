#!/usr/bin/env python3
"""
프로젝트 환경 설정 및 검증 통합 스크립트
- GPU/CUDA 환경 확인
- 라이브러리 버전 확인
- 데이터셋 로딩 테스트
- 모델 설정 검증
- 기본 매개변수 확인
"""

import os
import sys
import subprocess
import logging
import argparse
import inspect

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd):
    """명령어 실행 및 결과 반환"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_system_info():
    """시스템 정보 확인"""
    print("=== 시스템 정보 ===")
    stdout, _, _ = run_command("lsb_release -d")
    print(f"OS: {stdout}")
    stdout, _, _ = run_command("uname -r")
    print(f"Kernel: {stdout}")
    print(f"Python: {sys.version}")

def check_nvidia_driver():
    """NVIDIA 드라이버 확인"""
    print("\n=== NVIDIA 드라이버 확인 ===")
    stdout, stderr, code = run_command("nvidia-smi")
    if code == 0:
        print("✅ NVIDIA 드라이버 설치됨:")
        print(stdout)
        return True
    else:
        print("❌ NVIDIA 드라이버 없음 또는 오류:")
        print(stderr)
        return False

def check_cuda_runtime():
    """CUDA 런타임 확인"""
    print("\n=== CUDA 런타임 확인 ===")
    stdout, stderr, code = run_command("nvcc --version")
    if code == 0:
        print("✅ CUDA 컴파일러 설치됨:")
        print(stdout)
        return True
    else:
        print("❌ CUDA 컴파일러 없음:")
        print(stderr)
        return False

def check_cuda_env_vars():
    """CUDA 환경 변수 확인"""
    print("\n=== CUDA 환경 변수 확인 ===")
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    path = os.environ.get('PATH', '')
    ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    
    print(f"CUDA_HOME: {cuda_home}")
    print(f"CUDA_PATH: {cuda_path}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"CUDA in PATH: {'/cuda/' in path.lower()}")
    print(f"LD_LIBRARY_PATH: {ld_path}")

def check_pytorch():
    """PyTorch CUDA 지원 확인"""
    print("\n=== PyTorch CUDA 지원 확인 ===")
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ PyTorch에서 CUDA 사용 불가")
            
        # CUDA 빌드 정보 확인
        print(f"PyTorch CUDA 빌드: {torch.version.cuda}")
        print(f"PyTorch cuDNN 빌드: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        return True
        
    except ImportError:
        print("❌ PyTorch 설치되지 않음")
        return False
    except Exception as e:
        print(f"❌ PyTorch 확인 중 오류: {e}")
        return False

def check_transformers():
    """Transformers 버전 확인"""
    print("\n=== Transformers 버전 확인 ===")
    try:
        import transformers
        print(f"Transformers 버전: {transformers.__version__}")
        return True
    except ImportError:
        print("❌ Transformers 설치되지 않음")
        return False

def check_trl():
    """TRL 버전 및 API 확인"""
    print("\n=== TRL 버전 및 API 확인 ===")
    try:
        import trl
        print(f"TRL 버전: {trl.__version__}")
        
        # PPOTrainer 확인
        from trl import PPOTrainer, PPOConfig
        print(f"PPOTrainer 클래스: {PPOTrainer}")
        
        # 클래스 메소드 확인
        methods = [method for method in dir(PPOTrainer) if not method.startswith('_')]
        print("\nPPOTrainer 클래스 메소드들:")
        for method in sorted(methods)[:10]:  # 처음 10개만 표시
            print(f"  - {method}")
            
        # 중요한 메소드들 확인
        important_methods = ['step', 'train', 'update', 'train_step', 'ppo_step', 'compute_loss']
        print("\n중요한 메소드들 존재 여부:")
        for method in important_methods:
            exists = hasattr(PPOTrainer, method)
            print(f"  - {method}: {'✅' if exists else '❌'}")
            
        # PPOConfig 테스트
        print("\nPPOConfig 테스트:")
        try:
            config = PPOConfig()
            print("✅ 기본 PPOConfig 생성 성공")
        except Exception as e:
            print(f"❌ 기본 PPOConfig 생성 실패: {e}")
            
        try:
            config = PPOConfig(learning_rate=1e-5, batch_size=4, mini_batch_size=2)
            print("✅ 최소 파라미터 PPOConfig 생성 성공")
        except Exception as e:
            print(f"❌ 최소 파라미터 PPOConfig 생성 실패: {e}")
            
        return True
        
    except ImportError:
        print("❌ TRL 설치되지 않음")
        return False
    except Exception as e:
        print(f"❌ TRL 확인 중 오류: {e}")
        return False

def check_bitsandbytes():
    """bitsandbytes CUDA 지원 확인"""
    print("\n=== bitsandbytes CUDA 지원 확인 ===")
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes 버전: {bnb.__version__}")
        
        # CUDA 컴파일 정보 확인
        stdout, stderr, code = run_command("python -c \"import bitsandbytes; print('bitsandbytes 로드 성공')\"")
        if code == 0:
            print("✅ bitsandbytes CUDA 라이브러리 로드 성공")
        else:
            print("❌ bitsandbytes CUDA 라이브러리 로드 실패:")
            print(stderr)
            
        return True
        
    except ImportError:
        print("❌ bitsandbytes 설치되지 않음")
        return False
    except Exception as e:
        print(f"❌ bitsandbytes 확인 중 오류: {e}")
        return False

def check_datasets():
    """Datasets 라이브러리 및 데이터셋 로딩 테스트"""
    print("\n=== 데이터셋 라이브러리 확인 ===")
    try:
        from datasets import load_dataset
        print("✅ datasets 라이브러리 로드 성공")
        
        # 테스트 데이터셋 로딩
        print("\n=== 데이터셋 로딩 테스트 ===")
        configs_to_test = ['normal_instructions', 'tiny-textbooks', 'claude_evol']
        
        for config in configs_to_test:
            try:
                print(f"\n테스트 중: {config}")
                ds = load_dataset("maywell/korean_textbooks", config, split="train", trust_remote_code=True)
                print(f"✅ 데이터셋 크기: {len(ds)}")
                print(f"컬럼들: {ds.column_names}")
                print(f"첫 번째 샘플 키들: {list(ds[0].keys())}")
                break  # 첫 번째 성공한 config 사용
            except Exception as e:
                print(f"❌ 오류 발생 ({config}): {e}")
                continue
                
        return True
        
    except ImportError:
        print("❌ datasets 라이브러리 설치되지 않음")
        return False
    except Exception as e:
        print(f"❌ datasets 확인 중 오류: {e}")
        return False

def test_argparse():
    """argparse 기능 테스트"""
    print("\n=== Argparse 기능 테스트 ===")
    try:
        parser = argparse.ArgumentParser(description="PPO 파인튜닝 스크립트")
        parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b")
        parser.add_argument("--dataset_name", type=str, default="nsmc")
        parser.add_argument("--output_dir", type=str, default="my_korean_ppo_finetuned_model")
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--lora_r", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=16)
        parser.add_argument("--input_max_text_length", type=int, default=12)
        parser.add_argument("--max_ppo_steps", type=int, default=20)
        
        # 더미 인자로 테스트
        args = parser.parse_args([])  # 빈 리스트로 기본값 테스트
        print("✅ Argparse 테스트 성공!")
        print(f"기본 설정들: model_name={args.model_name}, batch_size={args.batch_size}")
        return True
        
    except Exception as e:
        print(f"❌ Argparse 테스트 실패: {e}")
        return False

def test_model_return_dict():
    """모델 return_dict 동작 테스트"""
    print("\n=== 모델 return_dict 동작 테스트 ===")
    try:
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from trl import AutoModelForCausalLMWithValueHead
        
        model_name = "EleutherAI/polyglot-ko-1.3b"
        
        print("토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("모델 로딩 (시간이 걸릴 수 있습니다)...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        print(f"모델 config return_dict 설정 전: {getattr(model.config, 'return_dict', 'Not set')}")
        
        # return_dict 설정
        model.config.return_dict = True
        print(f"모델 config return_dict 설정 후: {model.config.return_dict}")
        
        # 테스트 출력
        test_input = tokenizer("안녕하세요", return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}
        
        import torch
        with torch.no_grad():
            output = model(**test_input)
        
        print(f"출력 타입: {type(output)}")
        print(f"tuple 여부: {isinstance(output, tuple)}")
        print(f"logits 속성 보유: {hasattr(output, 'logits')}")
        
        if hasattr(output, 'logits'):
            print("✅ 모델이 올바르게 ModelOutput with logits 반환")
            return True
        else:
            print("❌ 모델이 tuple 또는 logits 없는 객체 반환")
            print(f"출력 내용: {output}")
            return False

    except Exception as e:
        print(f"❌ 모델 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 환경 검증 함수"""
    print("🔍 프로젝트 환경 설정 및 검증을 시작합니다...\n")
    
    results = []
    
    # 시스템 정보
    check_system_info()
    
    # 하드웨어 및 드라이버 확인
    results.append(("NVIDIA 드라이버", check_nvidia_driver()))
    results.append(("CUDA 런타임", check_cuda_runtime()))
    check_cuda_env_vars()
    
    # 라이브러리 확인
    results.append(("PyTorch", check_pytorch()))
    results.append(("Transformers", check_transformers()))
    results.append(("TRL", check_trl()))
    results.append(("bitsandbytes", check_bitsandbytes()))
    results.append(("datasets", check_datasets()))
    
    # 기능 테스트
    results.append(("Argparse", test_argparse()))
    results.append(("모델 return_dict", test_model_return_dict()))
    
    # 결과 요약
    print("\n" + "="*60)
    print("🎯 환경 검증 결과 요약")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체 결과: {passed}/{total} 통과 ({passed/total*100:.1f}%)")
    
    # 권장사항
    print("\n=== 권장사항 ===")
    if passed == total:
        print("🎉 모든 환경 검증을 통과했습니다! 프로젝트를 진행할 수 있습니다.")
    else:
        print("⚠️  일부 검증이 실패했습니다. 다음 항목들을 확인해주세요:")
        for name, result in results:
            if not result:
                print(f"  - {name} 설정 및 설치 확인")
        
        if not any(result for name, result in results if name in ["NVIDIA 드라이버", "CUDA 런타임"]):
            print("\n💡 NVIDIA/CUDA 설치 가이드:")
            print("  1. sudo apt update && sudo apt install nvidia-driver-xxx")
            print("  2. CUDA Toolkit 설치")
            print("  3. 재부팅 후 nvidia-smi 확인")
        
        if not any(result for name, result in results if name in ["PyTorch", "bitsandbytes"]):
            print("\n💡 PyTorch 재설치 권장:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()