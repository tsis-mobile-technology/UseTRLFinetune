#!/bin/bash
"""
GPU 및 CUDA 환경 수정 스크립트
"""

echo "=== GPU 환경 수정 시작 ==="

# 현재 디렉토리 확인
SCRIPT_DIR="/media/proidea/hdd/Programming/UseTRLFinetune"
cd "$SCRIPT_DIR"

# 가상환경 활성화
source korean-llm-env/bin/activate

echo "1. 현재 패키지 확인..."
echo "PyTorch 버전:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "PyTorch 확인 실패"

echo -e "\nbitsandbytes 버전:"
python -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null || echo "bitsandbytes 확인 실패"

echo -e "\n2. CUDA 관련 패키지 제거..."
pip uninstall torch torchvision torchaudio bitsandbytes -y

echo -e "\n3. 캐시 정리..."
pip cache purge
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "CUDA 캐시 정리 건너뜀"

echo -e "\n4. NVIDIA 드라이버 확인..."
nvidia-smi || echo "NVIDIA 드라이버 확인 실패"

echo -e "\n5. CUDA 버전 확인..."
nvcc --version || echo "CUDA 컴파일러 확인 실패"

# CUDA 버전 자동 감지
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "감지된 CUDA 버전: $CUDA_VERSION"
fi

echo -e "\n6. PyTorch GPU 버전 설치..."
if [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "CUDA 12.x 감지 - PyTorch CUDA 12.1 설치"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    echo "CUDA 11.x 감지 - PyTorch CUDA 11.8 설치"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA 버전 자동 감지 실패 - 최신 CUDA 12.1 버전으로 설치"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

echo -e "\n7. bitsandbytes GPU 버전 설치..."
# bitsandbytes 최신 버전 설치 (GPU 지원)
pip install bitsandbytes

echo -e "\n8. 설치 검증..."
python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 개수: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA 사용 불가')
"

echo -e "\n9. bitsandbytes 검증..."
python -c "
import bitsandbytes as bnb
print(f'bitsandbytes 버전: {bnb.__version__}')
try:
    import bitsandbytes.functional as F
    print('✅ bitsandbytes GPU 기능 로드 성공')
except Exception as e:
    print(f'❌ bitsandbytes GPU 기능 로드 실패: {e}')
"

echo -e "\n10. 환경 변수 설정..."
# CUDA 환경 변수 설정 (임시)
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME: $CUDA_HOME"

echo -e "\n=== GPU 환경 수정 완료 ==="
echo "다음 명령으로 재테스트:"
echo "python check_gpu.py"