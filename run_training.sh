#!/bin/bash
# Ollama GPT-OSS-20B Korean Fine-tuning 실행 스크립트

echo "🚀 Ollama GPT-OSS-20B Korean Fine-tuning 시작"

# 가상환경 활성화
source korean-llm-env/bin/activate

# GPU 메모리 정리
echo "🧹 GPU 메모리 정리 중..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU 메모리 정리 완료')"

# 데이터 확인
if [ ! -f "data/korean_wikipedia_data.jsonl" ]; then
    echo "📚 한국어 데이터 수집 중..."
    python scrape_wiki.py --max-articles 50 --delay 1.5
fi

# Fine-tuning 실행 옵션
EPOCHS=${1:-1}           # 첫 번째 인자 또는 기본값 1
BATCH_SIZE=${2:-2}       # 두 번째 인자 또는 기본값 2
LEARNING_RATE=${3:-2e-4} # 세 번째 인자 또는 기본값 2e-4

echo "📋 훈련 설정:"
echo "   - 에포크: $EPOCHS"
echo "   - 배치 크기: $BATCH_SIZE"  
echo "   - 학습률: $LEARNING_RATE"
echo ""

# Fine-tuning 실행
echo "🔥 Fine-tuning 실행 중..."
python ollama_gpt-oss-20b_unsloth.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --data-path data/korean_wikipedia_data.jsonl \
    --output-dir my_korean_gpt_oss_20b_lora

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Fine-tuning 완료!"
    echo "📁 모델 저장 위치: my_korean_gpt_oss_20b_lora/"
    echo "📄 로그 파일: training.log"
    echo ""
    echo "💡 다음 단계:"
    echo "   1. 추론 테스트: python ollama_gpt-oss-20b_unsloth.py --test-only"
    echo "   2. 더 많은 데이터로 재훈련: python scrape_wiki.py --max-articles 200"
    echo "   3. Ollama에서 사용하기 위해 GGUF 변환 필요"
else
    echo ""
    echo "❌ Fine-tuning 실패"
    echo "📄 로그 확인: cat training.log"
fi