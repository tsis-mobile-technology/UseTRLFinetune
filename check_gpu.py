#!/usr/bin/env python3
"""
GPU 및 CUDA 환경 확인 스크립트
"""
import subprocess
import sys
import os

def run_command(cmd):
    """명령어 실행 및 결과 반환"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_nvidia_driver():
    """NVIDIA 드라이버 확인"""
    print("=== NVIDIA 드라이버 확인 ===")
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
        
    except ImportError:
        print("❌ PyTorch 설치되지 않음")
    except Exception as e:
        print(f"❌ PyTorch 확인 중 오류: {e}")

def check_bitsandbytes():
    """bitsandbytes CUDA 지원 확인"""
    print("\n=== bitsandbytes CUDA 지원 확인 ===")
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes 버전: {bnb.__version__}")
        
        # CUDA 컴파일 정보 확인
        stdout, stderr, code = run_command("python -c \"import bitsandbytes; print(bitsandbytes.cuda_setup.common.get_cuda_lib_handle())\"")
        if code == 0:
            print("✅ bitsandbytes CUDA 라이브러리 로드 성공")
        else:
            print("❌ bitsandbytes CUDA 라이브러리 로드 실패:")
            print(stderr)
            
    except ImportError:
        print("❌ bitsandbytes 설치되지 않음")
    except Exception as e:
        print(f"❌ bitsandbytes 확인 중 오류: {e}")

def check_transformers():
    """transformers 버전 확인"""
    print("\n=== Transformers 버전 확인 ===")
    try:
        import transformers
        print(f"Transformers 버전: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 설치되지 않음")

def main():
    print("GPU 및 CUDA 환경 진단을 시작합니다...\n")
    
    # 시스템 정보
    print("=== 시스템 정보 ===")
    stdout, _, _ = run_command("lsb_release -d")
    print(f"OS: {stdout}")
    stdout, _, _ = run_command("uname -r")
    print(f"Kernel: {stdout}")
    
    # 각종 확인
    has_nvidia = check_nvidia_driver()
    has_cuda = check_cuda_runtime()
    check_cuda_env_vars()
    check_pytorch()
    check_bitsandbytes()
    check_transformers()
    
    # 권장사항
    print("\n=== 진단 결과 및 권장사항 ===")
    if not has_nvidia:
        print("❌ NVIDIA 드라이버가 설치되지 않았습니다.")
        print("   sudo apt update && sudo apt install nvidia-driver-xxx (xxx는 버전)")
    elif not has_cuda:
        print("❌ CUDA 런타임이 설치되지 않았습니다.")
        print("   CUDA Toolkit 설치 필요")
    else:
        print("✅ 기본 NVIDIA/CUDA 환경은 정상입니다.")
        print("   PyTorch 및 bitsandbytes 재설치를 권장합니다.")

if __name__ == "__main__":
    main()