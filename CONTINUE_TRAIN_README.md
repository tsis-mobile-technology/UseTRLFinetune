# 추가 파인튜닝 스크립트 (Continue Training)

이 스크립트는 기존에 파인튜닝된 모델에서 새로운 데이터로 추가 학습을 진행하는 두 가지 방법을 지원합니다.

## 지원하는 학습 방식

1. **기존 어댑터 계속 학습 (Continue Training)**
   - 기존 LoRA 어댑터에서 바로 계속 학습하는 방식
   - 가장 효율적이고 권장되는 방법
   - 기존 학습 결과를 유지하면서 새로운 데이터로 조정

2. **병합 후 새 어댑터 학습 (Merge and New Training)**
   - 기존 어댑터를 기본 모델과 병합한 후, 이를 새로운 기본 모델로 사용
   - 새 LoRA 어댑터를 처음부터 학습
   - 메모리가 더 많이 필요하지만 기존 학습과 새 학습을 명확히 분리하고 싶을 때 유용

## 사용 방법

### 1. 기존 어댑터 계속 학습

기존 파인튜닝된 모델(my_optimized_model 또는 my_korean_finetuned_model)에서 계속 학습:

```bash
python continue_train.py \
  --base-model EleutherAI/polyglot-ko-1.3b \
  --finetuned-model my_optimized_model \
  --data-path new_training_data.jsonl \
  --output-dir my_continued_model \
  --mode continue \
  --batch-size 4 \
  --epochs 3 \
  --use-8bit
```

### 2. 병합 후 새 어댑터 학습

기존 모델을 병합한 후 새 어댑터로 학습:

```bash
python continue_train.py \
  --base-model EleutherAI/polyglot-ko-1.3b \
  --finetuned-model my_optimized_model \
  --data-path new_training_data.jsonl \
  --output-dir my_merged_new_model \
  --mode merge_and_new \
  --batch-size 4 \
  --epochs 3 \
  --use-8bit
```

### 3. 학습 후 병합된 전체 모델 저장

학습 후 LoRA 어댑터와 기본 모델을 병합하여 완전한 모델로 저장:

```bash
python continue_train.py \
  --base-model EleutherAI/polyglot-ko-1.3b \
  --finetuned-model my_optimized_model \
  --data-path new_training_data.jsonl \
  --output-dir my_continued_model \
  --mode continue \
  --merge-and-save \
  --batch-size 4 \
  --epochs 3 \
  --use-8bit
```

## 주요 옵션 설명

| 옵션 | 설명 |
|------|------|
| `--base-model` | 기본 모델 이름 또는 경로 (기본값: EleutherAI/polyglot-ko-1.3b) |
| `--finetuned-model` | 기존에 파인튜닝된 모델 경로 (my_optimized_model 또는 my_korean_finetuned_model) |
| `--data-path` | 새로운 훈련 데이터 JSONL 파일 경로 (필수) |
| `--output-dir` | 모델 저장 디렉토리 (기본값: my_continued_model) |
| `--mode` | 학습 방식: continue(기존 어댑터 계속 학습) 또는 merge_and_new(병합 후 새 어댑터) |
| `--merge-and-save` | 학습 후 어댑터와 모델 병합해서 저장 (병합된 전체 모델) |
| `--batch-size` | 배치 크기 (기본값: 4) |
| `--epochs` | 훈련 에포크 수 (기본값: 3) |
| `--learning-rate` | 학습률 (기본값: 2e-4) |
| `--max-length` | 최대 시퀀스 길이 (기본값: 512) |
| `--limit-samples` | 훈련에 사용할 최대 샘플 수 (기본값: 1000, 메모리 제한) |
| `--lora-r` | LoRA r 값 (rank) (기본값: 64) |
| `--lora-alpha` | LoRA alpha 값 (기본값: 16) |
| `--lora-dropout` | LoRA dropout 비율 (기본값: 0.1) |
| `--use-8bit` | 8비트 양자화 사용 (메모리 절약) |
| `--use-4bit` | 4비트 양자화 사용 (더 많은 메모리 절약) |
| `--use-fp16` | FP16 훈련 사용 (속도 향상) |

## 학습 데이터 형식

학습 데이터는 다음과 같은 JSONL 형식이어야 합니다:

```json
{"text": "여기에 훈련 텍스트가 들어갑니다."}
{"text": "또 다른 훈련 텍스트 예시입니다."}
```

다른 필드 이름(content, dialogue 등)도 자동으로 인식됩니다.