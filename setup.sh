#!/bin/bash

# 한국어 LLM 파인튜닝 환경 설정 스크립트

echo "🚀 한국어 LLM 파인튜닝 환경 설정을 시작합니다..."

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수 정의
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 시스템 요구사항 확인
echo "🔍 시스템 요구사항 확인 중..."

# Python 버전 확인
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    print_status "Python 발견: $python_version"
else
    print_error "Python3가 설치되어 있지 않습니다."
    exit 1
fi

# NVIDIA GPU 확인
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_status "GPU 발견: $gpu_info"
else
    print_warning "NVIDIA GPU가 감지되지 않았습니다. CPU 모드로 실행될 수 있습니다."
fi

# 가상환경 확인 및 생성
if [ ! -d "korean-llm-env" ]; then
    echo "📦 가상환경을 생성합니다..."
    python3 -m venv korean-llm-env
    print_status "가상환경 생성 완료"
else
    print_status "기존 가상환경을 발견했습니다."
fi

# 가상환경 활성화
echo "🔄 가상환경을 활성화합니다..."
source korean-llm-env/bin/activate

if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "가상환경 활성화 완료: $VIRTUAL_ENV"
else
    print_error "가상환경 활성화에 실패했습니다."
    exit 1
fi

# Python 개발 헤더 확인 (Ubuntu/Debian)
if command -v apt &> /dev/null; then
    echo "🔧 시스템 의존성을 확인합니다..."
    
    python_dev_version=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}-dev')")
    
    if ! dpkg -l | grep -q $python_dev_version; then
        print_warning "$python_dev_version가 설치되어 있지 않습니다."
        echo "다음 명령어로 설치하세요:"
        echo "sudo apt update && sudo apt install $python_dev_version"
    else
        print_status "$python_dev_version가 이미 설치되어 있습니다."
    fi
fi

# pip 업그레이드
echo "📦 pip를 업그레이드합니다..."
pip install --upgrade pip
print_status "pip 업그레이드 완료"

# requirements.txt가 있으면 설치
if [ -f "requirements.txt" ]; then
    echo "📋 requirements.txt에서 패키지를 설치합니다..."
    pip install -r requirements.txt
    print_status "패키지 설치 완료"
else
    # 수동으로 필수 패키지 설치
    echo "📋 필수 패키지를 설치합니다..."
    pip install torch torchvision torchaudio transformers datasets accelerate trl peft bitsandbytes sentencepiece
    print_status "필수 패키지 설치 완료"
fi

# 설치된 패키지 확인
echo "🔍 설치된 주요 패키지 버전 확인..."
python3 -c "
import sys
packages = ['torch', 'transformers', 'datasets', 'accelerate', 'trl', 'peft', 'bitsandbytes']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: Not installed')
"

# GPU 사용 가능성 확인
echo "🔍 PyTorch GPU 지원 확인..."
python3 -c "
import torch
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 개수: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)')
"

# 디렉터리 구조 확인
echo "📁 프로젝트 구조 확인..."
required_files=("train.py" "test.py" "load.py" "inference.py" "evaluate.py")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file 존재"
    else
        print_warning "$file 없음"
    fi
done

if [ -d "my_korean_finetuned_model" ]; then
    print_status "파인튜닝된 모델 디렉터리 존재"
    model_size=$(du -sh my_korean_finetuned_model | cut -f1)
    echo "  모델 크기: $model_size"
else
    print_warning "파인튜닝된 모델이 없습니다. train.py를 실행하여 모델을 학습하세요."
fi

echo ""
echo "🎉 환경 설정이 완료되었습니다!"
echo ""
echo "📚 사용 방법:"
echo "  1. 가상환경 활성화: source korean-llm-env/bin/activate"
echo "  2. 모델 학습: python train.py"
echo "  3. 모델 테스트: python test.py"
echo "  4. 모델 평가: python evaluate.py"
echo "  5. 대화형 추론: python inference.py --interactive"
echo ""
echo "💡 팁: 가상환경을 비활성화하려면 'deactivate' 명령어를 사용하세요."