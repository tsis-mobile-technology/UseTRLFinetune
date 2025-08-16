#!/usr/bin/env python3
"""
개선된 GPU 설정 복구 및 진단 스크립트
"""
import subprocess
import sys
import os
import shutil

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
    print("=== 종합 시스템 진단 ===")
    
    # 1. NVIDIA 드라이버 확인
    print("\n1. NVIDIA 드라이버 확인:")
    stdout, stderr, code = run_cmd("nvidia-smi")
    if code == 0:
        print("✅ NVIDIA 드라이버 설치됨")
        print(stdout[:200] + "..." if len(stdout) > 200 else stdout)
        nvidia_available = True
    else:
        print("❌ NVIDIA 드라이버 없음 또는 오류")
        print(f"오류: {stderr}")
        nvidia_available = False
    
    # 2. CUDA Toolkit 확인
    print("\n2. CUDA Toolkit 확인:")
    stdout, stderr, code = run_cmd("nvcc --version")
    if code == 0:
        print("✅ CUDA 컴파일러 설치됨")
        print(stdout)
        cuda_toolkit = True
    else:
        print("❌ CUDA 컴파일러 없음")
        print(f"오류: {stderr}")
        cuda_toolkit = False
    
    # 3. CUDA 라이브러리 경로 확인
    print("\n3. CUDA 라이브러리 확인:")
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
            print(f"✅ CUDA 경로 발견: {path}")
            lib_path = os.path.join(path, "lib64")
            if os.path.exists(lib_path):
                print(f"  - 라이브러리 경로: {lib_path}")
            found_cuda = True
            break
    
    if not found_cuda:
        print("❌ CUDA 설치 경로를 찾을 수 없음")
    
    # 4. 환경 변수 확인
    print("\n4. 환경 변수 확인:")
    env_vars = {
        'CUDA_HOME': os.environ.get('CUDA_HOME', 'Not set'),
        'CUDA_PATH': os.environ.get('CUDA_PATH', 'Not set'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', 'Not set')[:100] + "..." if len(os.environ.get('LD_LIBRARY_PATH', '')) > 100 else os.environ.get('LD_LIBRARY_PATH', 'Not set')
    }
    
    for var, value in env_vars.items():
        print(f"  {var}: {value}")
    
    # 5. GPU 디바이스 확인
    print("\n5. GPU 디바이스 확인:")
    stdout, stderr, code = run_cmd("lspci | grep -i nvidia")
    if code == 0 and stdout:
        print("✅ NVIDIA GPU 발견:")
        print(stdout)
        gpu_hardware = True
    else:
        print("❌ NVIDIA GPU를 찾을 수 없음")
        gpu_hardware = False
    
    # 6. 커널 모듈 확인
    print("\n6. NVIDIA 커널 모듈 확인:")
    stdout, stderr, code = run_cmd("lsmod | grep nvidia")
    if code == 0 and stdout:
        print("✅ NVIDIA 커널 모듈 로드됨:")
        print(stdout)
        kernel_module = True
    else:
        print("❌ NVIDIA 커널 모듈이 로드되지 않음")
        kernel_module = False
    
    return {
        'nvidia_driver': nvidia_available,
        'cuda_toolkit': cuda_toolkit,
        'cuda_libs': found_cuda,
        'gpu_hardware': gpu_hardware,
        'kernel_module': kernel_module
    }

def install_pytorch_gpu():
    """발전된 PyTorch GPU 버전 설치"""
    print("\nPyTorch GPU 버전 설치 중...")
    
    # 기존 torch 제거 (오류 무시)
    print("  기존 PyTorch 제거 중...")
    run_cmd([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"], shell=False)
    
    # 캐시 정리
    run_cmd([sys.executable, "-m", "pip", "cache", "purge"], shell=False)
    
    # PyTorch CUDA 버전 설치 (여러 버전 시도)
    cuda_versions = [
        ("cu121", "https://download.pytorch.org/whl/cu121"),
        ("cu118", "https://download.pytorch.org/whl/cu118"),
        ("cpu", "https://download.pytorch.org/whl/cpu")  # 마지막 대안
    ]
    
    for version, url in cuda_versions:
        print(f"  PyTorch {version} 버전 설치 시도...")
        stdout, stderr, code = run_cmd([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", url
        ], shell=False)
        
        if code == 0:
            print(f"  ✅ PyTorch {version} 설치 성공")
            return version
        else:
            print(f"  ❌ PyTorch {version} 설치 실패: {stderr[:100]}...")
    
    print("  ❌ 모든 PyTorch 버전 설치 실패")
    return None

def install_bitsandbytes_gpu():
    """향상된 bitsandbytes GPU 버전 설치"""
    print("\nbitsandbytes GPU 버전 설치 중...")
    
    # 기존 bitsandbytes 제거
    print("  기존 bitsandbytes 제거 중...")
    run_cmd([sys.executable, "-m", "pip", "uninstall", "bitsandbytes", "-y"], shell=False)
    
    # 여러 방법으로 설치 시도
    install_methods = [
        # 방법 1: 최신 버전
        ([sys.executable, "-m", "pip", "install", "bitsandbytes"], "최신 버전"),
        # 방법 2: 특정 버전
        ([sys.executable, "-m", "pip", "install", "bitsandbytes==0.41.1"], "버전 0.41.1"),
        # 방법 3: 소스에서 컴파일
        ([sys.executable, "-m", "pip", "install", "bitsandbytes", "--no-binary", "bitsandbytes"], "소스 컴파일"),
        # 방법 4: CPU 전용 버전 (마지막 대안)
        ([sys.executable, "-m", "pip", "install", "bitsandbytes", "--force-reinstall"], "강제 재설치")
    ]
    
    for cmd, desc in install_methods:
        print(f"  {desc} 설치 시도...")
        stdout, stderr, code = run_cmd(cmd, shell=False)
        
        if code == 0:
            print(f"  ✅ bitsandbytes {desc} 설치 성공")
            return True
        else:
            print(f"  ❌ {desc} 설치 실패: {stderr[:100]}...")
    
    print("  ⚠️ 모든 bitsandbytes 설치 방법 실패 - CPU 모드에서만 사용 가능")
    return False

def test_gpu_setup():
    """향상된 GPU 설정 테스트"""
    print("\nGPU 설정 테스트 중...")
    
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"cuDNN 버전: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                # GPU 메모리 정보
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  - 총 메모리: {memory_total:.1f} GB")
            
            # 간단한 GPU 연산 테스트
            try:
                device = torch.device("cuda:0")
                x = torch.randn(100, 100).to(device)
                y = torch.randn(100, 100).to(device)
                z = torch.matmul(x, y)
                print("✅ GPU 연산 테스트 성공")
                return True
            except Exception as e:
                print(f"❌ GPU 연산 테스트 실패: {e}")
                return False
        else:
            print("❌ CUDA 사용 불가 - CPU 모드로 작동")
            
            # CPU 테스트
            try:
                x = torch.randn(100, 100)
                y = torch.randn(100, 100)
                z = torch.matmul(x, y)
                print("✅ CPU 연산 테스트 성공")
                return 'cpu'
            except Exception as e:
                print(f"❌ CPU 연산 테스트도 실패: {e}")
                return False
            
    except Exception as e:
        print(f"❌ PyTorch 테스트 실패: {e}")
        return False

def test_bitsandbytes():
    """향상된 bitsandbytes 테스트"""
    print("\nbitsandbytes 테스트 중...")
    
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes 버전: {bnb.__version__}")
        
        # GPU 지원 확인
        try:
            import bitsandbytes.functional as F
            # 간단한 양자화 테스트
            import torch
            if torch.cuda.is_available():
                x = torch.randn(10, 10).cuda()
                x_int8 = F.quantize_blockwise(x)
                print("✅ bitsandbytes GPU 기능 테스트 성공")
                return True
            else:
                print("⚠️ GPU 사용 불가로 bitsandbytes CPU 모드")
                return 'cpu'
        except Exception as e:
            print(f"⚠️ bitsandbytes GPU 기능 제한: {e}")
            print("  CPU 모드에서만 사용 가능")
            return 'cpu'
        
    except ImportError:
        print("❌ bitsandbytes 가져오기 실패")
        return False
    except Exception as e:
        print(f"❌ bitsandbytes 테스트 실패: {e}")
        return False

def set_environment_variables():
    """향상된 환경 변수 설정"""
    print("\n환경 변수 설정 중...")
    
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
        
        print(f"CUDA_HOME 설정: {cuda_home}")
        print(f"LD_LIBRARY_PATH 업데이트: {lib_path}")
    else:
        print("⚠️ CUDA 설치 경로를 찾을 수 없음")
    
    # GPU 가시성 설정
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("CUDA_VISIBLE_DEVICES=0 설정")
    
    # 추가 설정
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    print("메모리 최적화 설정 적용")

def generate_cpu_training_script():
    """기존 훈련 스크립트의 CPU 버전 생성"""
    print("\nCPU 모드 훈련 스크립트 생성 중...")
    
    cpu_script = '''
#!/bin/bash
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
    
    with open('/media/proidea/hdd/Programming/UseTRLFinetune/run_cpu_train.sh', 'w', encoding='utf-8') as f:
        f.write(cpu_script)
    
    # 실행 권한 부여
    os.chmod('/media/proidea/hdd/Programming/UseTRLFinetune/run_cpu_train.sh', 0o755)
    
    print("✅ CPU 모드 훈련 스크립트 생성: run_cpu_train.sh")

def provide_recommendations(system_status, pytorch_result, bnb_result):
    """상황에 맞는 추천사항 제공"""
    print("\n=== 추천사항 ===")
    
    if not system_status['nvidia_driver']:
        print("📝 NVIDIA 드라이버 설치 방법:")
        print("  sudo apt update")
        print("  sudo apt install nvidia-driver-535  # 또는 최신 버전")
        print("  sudo reboot")
        
    elif not system_status['cuda_toolkit']:
        print("📝 CUDA Toolkit 설치 방법:")
        print("  wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run")
        print("  sudo sh cuda_12.1.0_530.30.02_linux.run")
        
    elif not system_status['kernel_module']:
        print("📝 NVIDIA 커널 모듈 로드:")
        print("  sudo modprobe nvidia")
        print("  sudo nvidia-modprobe")
        
    # 훈련 방법 추천
    if pytorch_result == True and bnb_result == True:
        print("\n🎉 GPU 환경 완전 복구! 원래 명령어로 실행 가능:")
        print("  python continue_train.py --base-model EleutherAI/polyglot-ko-1.3b ...")
        
    elif pytorch_result == True and bnb_result in ['cpu', False]:
        print("\n🚀 PyTorch GPU 사용 가능, bitsandbytes 제한:")
        print("  --use-8bit 옵션을 제거하고 실행:")
        print("  python continue_train.py --base-model EleutherAI/polyglot-ko-1.3b --finetuned-model my_optimized_model --data-path web_data_add1.jsonl --output-dir my_continued_model --mode continue --batch-size 2 --lora-r 64 --lora-alpha 128 --max-length 1024 --epochs 10")
        
    else:
        print("\n💻 CPU 모드 추천 (메모리 제한에 맞춘 설정):")
        print("  ./run_cpu_train.sh")
        print("  또는 직접 실행:")
        print("  python continue_train.py --base-model EleutherAI/polyglot-ko-1.3b --finetuned-model my_optimized_model --data-path web_data_add1.jsonl --output-dir my_continued_model_cpu --mode continue --batch-size 1 --lora-r 32 --lora-alpha 64 --max-length 512 --epochs 5")
    
    print("\n🛠️ 추가 도구:")
    print("  - GPU 상태 확인: python check_gpu.py")
    print("  - 시스템 모니터링: nvidia-smi")
    print("  - 메모리 사용량: watch -n 1 'free -h && nvidia-smi'")

def main():
    print("=== 개선된 GPU 환경 진단 및 복구 ===\n")
    
    # 1단계: 종합 시스템 진단
    system_status = comprehensive_system_check()
    
    # 2단계: 환경 변수 설정
    set_environment_variables()
    
    # 3단계: PyTorch 설치
    pytorch_version = install_pytorch_gpu()
    if not pytorch_version:
        print("\n❌ PyTorch 설치 실패 - 계속 진행할 수 없습니다.")
        return
    
    # 4단계: bitsandbytes 설치
    bnb_ok = install_bitsandbytes_gpu()
    
    # 5단계: 설치 검증
    print("\n=== 설치 검증 ===")
    pytorch_result = test_gpu_setup()
    bnb_result = test_bitsandbytes()
    
    # 6단계: CPU 모드 스크립트 생성
    generate_cpu_training_script()
    
    # 7단계: 추천사항 제공
    provide_recommendations(system_status, pytorch_result, bnb_result)
    
    # 최종 결과
    print("\n=== 최종 결과 ===")
    if pytorch_result == True and bnb_result == True:
        print("✅ GPU 환경 완전 복구!")
        print("원래 명령어로 훈련을 실행할 수 있습니다.")
    elif pytorch_result == True:
        print("✅ PyTorch GPU 사용 가능")
        print("⚠️ bitsandbytes 제한 - --use-8bit 옵션 제거 후 실행")
    elif pytorch_result == 'cpu':
        print("⚠️ CPU 모드로만 작동")
        print("GPU 문제로 인해 CPU로 훈련을 진행합니다.")
    else:
        print("❌ 복구 실패")
        print("시스템 CUDA 설정을 확인해주세요.")
    
    print("\n🛠️ 다음 단계:")
    print("1. 상황에 맞는 명령어로 훈련 실행")
    print("2. 문제 지속 시: python check_gpu.py로 재진단")
    print("3. 시스템 재부팅 후 다시 시도")

if __name__ == "__main__":
    main()