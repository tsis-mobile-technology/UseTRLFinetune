#!/usr/bin/env python3
"""
통합 환경 수정 도구
기존의 fix_gpu.sh와 fix_gpu_env.py를 하나로 통합
"""

import os
import sys
import subprocess
import argparse
import logging
import shutil

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 경로
PROJECT_ROOT = "/media/proidea/hdd/Programming/UseTRLFinetune"
VENV_PATH = "/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env"
PYTHON_PATH = f"{VENV_PATH}/bin/python"

def run_cmd(cmd, shell=True, check=False):
    """명령 실행 (오류 시에도 계속 진행)"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return "", e.stderr, e.returncode
    except Exception as e:
        return "", str(e), 1

def comprehensive_system_check():
    """종합적인 시스템 GPU/CUDA 진단"""
    logger.info("=== 종합 시스템 진단 ===")
    
    # 1. NVIDIA 드라이버 확인
    logger.info("\n1. NVIDIA 드라이버 확인:")
    stdout, stderr, code = run_cmd("nvidia-smi")
    if code == 0:
        logger.info("✅ NVIDIA 드라이버 설치됨")
        logger.info(stdout[:200] + "..." if len(stdout) > 200 else stdout)
        nvidia_available = True
    else:
        logger.error("❌ NVIDIA 드라이버 없음 또는 오류")
        logger.error(f"오류: {stderr}")
        nvidia_available = False
    
    # 2. CUDA Toolkit 확인
    logger.info("\n2. CUDA Toolkit 확인:")
    stdout, stderr, code = run_cmd("nvcc --version")
    if code == 0:
        logger.info("✅ CUDA 컴파일러 설치됨")
        logger.info(stdout)
        cuda_toolkit = True
    else:
        logger.warning("❌ CUDA 컴파일러 없음")
        logger.warning(f"오류: {stderr}")
        cuda_toolkit = False
    
    # 3. CUDA 라이브러리 경로 확인
    logger.info("\n3. CUDA 라이브러리 확인:")
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-12.0",
        "/opt/cuda"
    ]
    
    found_cuda = False
    for path in cuda_paths:
        if os.path.exists(path):
            logger.info(f"✅ CUDA 경로 발견: {path}")
            lib_path = os.path.join(path, "lib64")
            if os.path.exists(lib_path):
                logger.info(f"  - 라이브러리 경로: {lib_path}")
            found_cuda = True
            break
    
    if not found_cuda:
        logger.warning("❌ CUDA 설치 경로를 찾을 수 없음")
    
    # 4. 환경 변수 확인
    logger.info("\n4. 환경 변수 확인:")
    env_vars = {
        'CUDA_HOME': os.environ.get('CUDA_HOME', 'Not set'),
        'CUDA_PATH': os.environ.get('CUDA_PATH', 'Not set'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', 'Not set')[:100] + "..." if len(os.environ.get('LD_LIBRARY_PATH', '')) > 100 else os.environ.get('LD_LIBRARY_PATH', 'Not set')
    }
    
    for var, value in env_vars.items():
        logger.info(f"  {var}: {value}")
    
    # 5. GPU 디바이스 확인
    logger.info("\n5. GPU 디바이스 확인:")
    stdout, stderr, code = run_cmd("lspci | grep -i nvidia")
    if code == 0 and stdout:
        logger.info("✅ NVIDIA GPU 발견:")
        logger.info(stdout)
        gpu_hardware = True
    else:
        logger.warning("❌ NVIDIA GPU를 찾을 수 없음")
        gpu_hardware = False
    
    # 6. 커널 모듈 확인
    logger.info("\n6. NVIDIA 커널 모듈 확인:")
    stdout, stderr, code = run_cmd("lsmod | grep nvidia")
    if code == 0 and stdout:
        logger.info("✅ NVIDIA 커널 모듈 로드됨:")
        logger.info(stdout)
        kernel_module = True
    else:
        logger.warning("❌ NVIDIA 커널 모듈이 로드되지 않음")
        kernel_module = False
    
    return {
        'nvidia_driver': nvidia_available,
        'cuda_toolkit': cuda_toolkit,
        'cuda_libs': found_cuda,
        'gpu_hardware': gpu_hardware,
        'kernel_module': kernel_module
    }

def fix_gpu_environment():
    """GPU 환경 수정 (기존 fix_gpu.sh 기능)"""
    logger.info("=== GPU 환경 수정 시작 ===")
    
    # 현재 디렉토리 확인
    os.chdir(PROJECT_ROOT)
    
    logger.info("1. 현재 패키지 확인...")
    logger.info("PyTorch 버전:")
    stdout, stderr, code = run_cmd(f"{PYTHON_PATH} -c \"import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())\"")
    if code == 0:
        logger.info(stdout)
    else:
        logger.warning("PyTorch 확인 실패")
    
    logger.info("\nbitsandbytes 버전:")
    stdout, stderr, code = run_cmd(f"{PYTHON_PATH} -c \"import bitsandbytes; print(bitsandbytes.__version__)\"")
    if code == 0:
        logger.info(stdout)
    else:
        logger.warning("bitsandbytes 확인 실패")
    
    logger.info("\n2. CUDA 관련 패키지 제거...")
    packages_to_remove = ["torch", "torchvision", "torchaudio", "bitsandbytes"]
    for package in packages_to_remove:
        run_cmd(f"{PYTHON_PATH} -m pip uninstall {package} -y")
    
    logger.info("\n3. 캐시 정리...")
    run_cmd(f"{PYTHON_PATH} -m pip cache purge")
    
    logger.info("\n4. NVIDIA 드라이버 확인...")
    stdout, stderr, code = run_cmd("nvidia-smi")
    if code == 0:
        logger.info("✅ NVIDIA 드라이버 정상")
    else:
        logger.warning("❌ NVIDIA 드라이버 확인 실패")
    
    logger.info("\n5. CUDA 버전 확인...")
    stdout, stderr, code = run_cmd("nvcc --version")
    
    # CUDA 버전 자동 감지
    cuda_version = ""
    if code == 0:
        for line in stdout.split('\n'):
            if 'release' in line:
                import re
                match = re.search(r'release (\d+\.\d+)', line)
                if match:
                    cuda_version = match.group(1)
                    logger.info(f"감지된 CUDA 버전: {cuda_version}")
                    break
    
    logger.info("\n6. PyTorch GPU 버전 설치...")
    if cuda_version.startswith("12."):
        logger.info("CUDA 12.x 감지 - PyTorch CUDA 12.1 설치")
        torch_url = "https://download.pytorch.org/whl/cu121"
    elif cuda_version.startswith("11."):
        logger.info("CUDA 11.x 감지 - PyTorch CUDA 11.8 설치")
        torch_url = "https://download.pytorch.org/whl/cu118"
    else:
        logger.info("CUDA 버전 자동 감지 실패 - 최신 CUDA 12.1 버전으로 설치")
        torch_url = "https://download.pytorch.org/whl/cu121"
    
    stdout, stderr, code = run_cmd(f"{PYTHON_PATH} -m pip install torch torchvision torchaudio --index-url {torch_url}")
    if code == 0:
        logger.info("✅ PyTorch 설치 성공")
    else:
        logger.error(f"❌ PyTorch 설치 실패: {stderr}")
    
    logger.info("\n7. bitsandbytes GPU 버전 설치...")
    stdout, stderr, code = run_cmd(f"{PYTHON_PATH} -m pip install bitsandbytes")
    if code == 0:
        logger.info("✅ bitsandbytes 설치 성공")
    else:
        logger.error(f"❌ bitsandbytes 설치 실패: {stderr}")
    
    # 설치 검증
    verify_installation()

def install_pytorch_gpu():
    """발전된 PyTorch GPU 버전 설치"""
    logger.info("\nPyTorch GPU 버전 설치 중...")
    
    # 기존 torch 제거
    logger.info("  기존 PyTorch 제거 중...")
    run_cmd(f"{PYTHON_PATH} -m pip uninstall torch torchvision torchaudio -y")
    
    # 캐시 정리
    run_cmd(f"{PYTHON_PATH} -m pip cache purge")
    
    # PyTorch CUDA 버전 설치 (여러 버전 시도)
    cuda_versions = [
        ("cu121", "https://download.pytorch.org/whl/cu121"),
        ("cu118", "https://download.pytorch.org/whl/cu118"),
        ("cpu", "https://download.pytorch.org/whl/cpu")  # 마지막 대안
    ]
    
    for version, url in cuda_versions:
        logger.info(f"  PyTorch {version} 버전 설치 시도...")
        stdout, stderr, code = run_cmd(f"{PYTHON_PATH} -m pip install torch torchvision torchaudio --index-url {url}")
        
        if code == 0:
            logger.info(f"  ✅ PyTorch {version} 설치 성공")
            return version
        else:
            logger.warning(f"  ❌ PyTorch {version} 설치 실패: {stderr[:100]}...")
    
    logger.error("  ❌ 모든 PyTorch 버전 설치 실패")
    return None

def install_bitsandbytes_gpu():
    """향상된 bitsandbytes GPU 버전 설치"""
    logger.info("\nbitsandbytes GPU 버전 설치 중...")
    
    # 기존 bitsandbytes 제거
    logger.info("  기존 bitsandbytes 제거 중...")
    run_cmd(f"{PYTHON_PATH} -m pip uninstall bitsandbytes -y")
    
    # 여러 방법으로 설치 시도
    install_methods = [
        (f"{PYTHON_PATH} -m pip install bitsandbytes", "최신 버전"),
        (f"{PYTHON_PATH} -m pip install bitsandbytes==0.41.1", "버전 0.41.1"),
        (f"{PYTHON_PATH} -m pip install bitsandbytes --no-binary bitsandbytes", "소스 컴파일"),
        (f"{PYTHON_PATH} -m pip install bitsandbytes --force-reinstall", "강제 재설치")
    ]
    
    for cmd, desc in install_methods:
        logger.info(f"  {desc} 설치 시도...")
        stdout, stderr, code = run_cmd(cmd)
        
        if code == 0:
            logger.info(f"  ✅ bitsandbytes {desc} 설치 성공")
            return True
        else:
            logger.warning(f"  ❌ {desc} 설치 실패: {stderr[:100]}...")
    
    logger.warning("  ⚠️ 모든 bitsandbytes 설치 방법 실패 - CPU 모드에서만 사용 가능")
    return False

def verify_installation():
    """설치 검증"""
    logger.info("\n8. 설치 검증...")
    
    # PyTorch 검증
    pytorch_test = f"""
import torch
print(f'PyTorch 버전: {{torch.__version__}}')
print(f'CUDA 사용 가능: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {{torch.version.cuda}}')
    print(f'cuDNN 버전: {{torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}}')
    print(f'GPU 개수: {{torch.cuda.device_count()}}')
    
    for i in range(torch.cuda.device_count()):
        print(f'GPU {{i}}: {{torch.cuda.get_device_name(i)}}')
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  - 총 메모리: {{memory_total:.1f}} GB')
else:
    print('❌ CUDA 사용 불가')
"""
    
    stdout, stderr, code = run_cmd(f'{PYTHON_PATH} -c "{pytorch_test}"')
    if code == 0:
        logger.info("PyTorch 검증 결과:")
        logger.info(stdout)
    else:
        logger.error(f"PyTorch 검증 실패: {stderr}")
    
    # bitsandbytes 검증
    bnb_test = f"""
import bitsandbytes as bnb
print(f'bitsandbytes 버전: {{bnb.__version__}}')
try:
    import bitsandbytes.functional as F
    print('✅ bitsandbytes GPU 기능 로드 성공')
except Exception as e:
    print(f'❌ bitsandbytes GPU 기능 로드 실패: {{e}}')
"""
    
    stdout, stderr, code = run_cmd(f'{PYTHON_PATH} -c "{bnb_test}"')
    if code == 0:
        logger.info("bitsandbytes 검증 결과:")
        logger.info(stdout)
    else:
        logger.error(f"bitsandbytes 검증 실패: {stderr}")

def set_environment_variables():
    """향상된 환경 변수 설정"""
    logger.info("\n환경 변수 설정 중...")
    
    # CUDA 관련 환경 변수
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-12.0",
        "/opt/cuda"
    ]
    
    cuda_home = None
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_home = path
            break
    
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["CUDA_PATH"] = cuda_home
        
        # LD_LIBRARY_PATH 업데이트
        lib_path = os.path.join(cuda_home, "lib64")
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if lib_path not in current_ld_path:
            new_ld_path = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path
            os.environ["LD_LIBRARY_PATH"] = new_ld_path
        
        # PATH 업데이트
        bin_path = os.path.join(cuda_home, "bin")
        current_path = os.environ.get('PATH', '')
        if bin_path not in current_path:
            new_path = f"{bin_path}:{current_path}" if current_path else bin_path
            os.environ["PATH"] = new_path
        
        logger.info(f"CUDA_HOME 설정: {cuda_home}")
        logger.info(f"LD_LIBRARY_PATH 업데이트: {lib_path}")
    else:
        logger.warning("⚠️ CUDA 설치 경로를 찾을 수 없음")
    
    # GPU 가시성 설정
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.info("CUDA_VISIBLE_DEVICES=0 설정")
    
    # 추가 설정
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    logger.info("메모리 최적화 설정 적용")

def generate_cpu_training_script():
    """CPU 모드 훈련 스크립트 생성"""
    logger.info("\nCPU 모드 훈련 스크립트 생성 중...")
    
    cpu_script = '''#!/bin/bash
# GPU 문제 시 CPU 모드로 훈련을 실행하는 스크립트

echo "CPU 모드 훈련 시작..."

# 기본 매개변수 (메모리 제한에 맞춘 조정)
python continue_train.py \\
  --base-model EleutherAI/polyglot-ko-1.3b \\
  --finetuned-model my_optimized_model \\
  --data-path web_data_add1.jsonl \\
  --output-dir my_continued_model_cpu \\
  --mode continue \\
  --batch-size 1 \\
  --lora-r 32 \\
  --lora-alpha 64 \\
  --max-length 512 \\
  --epochs 5
  # --use-8bit 옵션 제거 (CPU에서 지원 안 됨)
  
echo "CPU 모드 훈련 완료"
'''
    
    cpu_script_path = os.path.join(PROJECT_ROOT, 'run_cpu_train_backup.sh')
    with open(cpu_script_path, 'w', encoding='utf-8') as f:
        f.write(cpu_script)
    
    # 실행 권한 부여
    os.chmod(cpu_script_path, 0o755)
    
    logger.info(f"✅ CPU 모드 훈련 스크립트 생성: {cpu_script_path}")

def provide_recommendations(system_status):
    """상황에 맞는 추천사항 제공"""
    logger.info("\n=== 추천사항 ===")
    
    if not system_status['nvidia_driver']:
        logger.info("📝 NVIDIA 드라이버 설치 방법:")
        logger.info("  sudo apt update")
        logger.info("  sudo apt install nvidia-driver-535  # 또는 최신 버전")
        logger.info("  sudo reboot")
        
    elif not system_status['cuda_toolkit']:
        logger.info("📝 CUDA Toolkit 설치 방법:")
        logger.info("  wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run")
        logger.info("  sudo sh cuda_12.1.0_530.30.02_linux.run")
        
    elif not system_status['kernel_module']:
        logger.info("📝 NVIDIA 커널 모듈 로드:")
        logger.info("  sudo modprobe nvidia")
        logger.info("  sudo nvidia-modprobe")
    
    logger.info("\n🛠️ 추가 도구:")
    logger.info("  - GPU 상태 확인: nvidia-smi")
    logger.info("  - 시스템 모니터링: watch -n 1 'free -h && nvidia-smi'")
    logger.info("  - 통합 실행 스크립트: python run_scripts.py list")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 환경 수정 도구")
    parser.add_argument(
        "command",
        choices=["diagnose", "fix-gpu", "install-pytorch", "install-bnb", "set-env", "verify", "full-fix", "list"],
        help="실행할 명령"
    )
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("\n=== 사용 가능한 명령어 ===")
        commands = {
            "diagnose": "시스템 GPU/CUDA 환경 진단",
            "fix-gpu": "GPU 환경 전체 수정 (기존 fix_gpu.sh)",
            "install-pytorch": "PyTorch GPU 버전 설치",
            "install-bnb": "bitsandbytes GPU 버전 설치",
            "set-env": "환경 변수 설정",
            "verify": "설치 검증",
            "full-fix": "전체 환경 복구 (추천)",
            "list": "이 도움말 출력"
        }
        
        for cmd, desc in commands.items():
            print(f"  {cmd:15} : {desc}")
        
        print(f"\n사용법: python {sys.argv[0]} <command>")
        return
    
    logger.info("=== 통합 환경 수정 도구 ===")
    
    if args.command == "diagnose":
        system_status = comprehensive_system_check()
        provide_recommendations(system_status)
        
    elif args.command == "fix-gpu":
        fix_gpu_environment()
        
    elif args.command == "install-pytorch":
        pytorch_version = install_pytorch_gpu()
        if pytorch_version:
            logger.info(f"✅ PyTorch {pytorch_version} 설치 완료")
        
    elif args.command == "install-bnb":
        bnb_ok = install_bitsandbytes_gpu()
        if bnb_ok:
            logger.info("✅ bitsandbytes 설치 완료")
        
    elif args.command == "set-env":
        set_environment_variables()
        
    elif args.command == "verify":
        verify_installation()
        
    elif args.command == "full-fix":
        logger.info("전체 환경 복구를 시작합니다...")
        
        # 1단계: 종합 시스템 진단
        system_status = comprehensive_system_check()
        
        # 2단계: 환경 변수 설정
        set_environment_variables()
        
        # 3단계: PyTorch 설치
        pytorch_version = install_pytorch_gpu()
        if not pytorch_version:
            logger.error("❌ PyTorch 설치 실패 - 복구를 중단합니다.")
            return
        
        # 4단계: bitsandbytes 설치
        bnb_ok = install_bitsandbytes_gpu()
        
        # 5단계: 설치 검증
        logger.info("\n=== 설치 검증 ===")
        verify_installation()
        
        # 6단계: CPU 모드 스크립트 생성
        generate_cpu_training_script()
        
        # 7단계: 추천사항 제공
        provide_recommendations(system_status)
        
        # 최종 결과
        logger.info("\n=== 최종 결과 ===")
        
        # PyTorch 상태 재확인
        stdout, stderr, code = run_cmd(f'{PYTHON_PATH} -c "import torch; print(torch.cuda.is_available())"')
        pytorch_gpu = code == 0 and "True" in stdout
        
        # bitsandbytes 상태 재확인  
        stdout, stderr, code = run_cmd(f'{PYTHON_PATH} -c "import bitsandbytes.functional"')
        bnb_gpu = code == 0
        
        if pytorch_gpu and bnb_gpu:
            logger.info("✅ GPU 환경 완전 복구!")
            logger.info("원래 명령어로 훈련을 실행할 수 있습니다.")
        elif pytorch_gpu:
            logger.info("✅ PyTorch GPU 사용 가능")
            logger.warning("⚠️ bitsandbytes 제한 - --use-8bit 옵션 제거 후 실행")
        else:
            logger.warning("⚠️ CPU 모드로만 작동")
            logger.info("GPU 문제로 인해 CPU로 훈련을 진행합니다.")
        
        logger.info("\n🛠️ 다음 단계:")
        logger.info("1. python run_scripts.py list로 사용 가능한 실행 옵션 확인")
        logger.info("2. 문제 지속 시 시스템 재부팅 후 다시 시도")

if __name__ == "__main__":
    main()