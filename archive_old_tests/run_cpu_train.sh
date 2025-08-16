
#!/bin/bash
# GPU 문제 시 CPU 모드로 훈련을 실행하는 스크립트

echo "CPU 모드 훈련 시작..."

# 기본 매개변수 (메모리 제한에 맞춘 조정)
python continue_train.py \
  --base-model EleutherAI/polyglot-ko-1.3b \
  --finetuned-model my_optimized_model \
  --data-path web_data_add1.jsonl \
  --output-dir my_continued_model_cpu \
  --mode continue \
  --batch-size 1 \
  --lora-r 32 \
  --lora-alpha 64 \
  --max-length 512 \
  --epochs 5
  # --use-8bit 옵션 제거 (CPU에서 지원 안 됨)
  
echo "CPU 모드 훈련 완료"
