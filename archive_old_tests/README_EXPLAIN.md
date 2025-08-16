# 훈련 스크립트 비교 및 설정 가이드

## 1. train.py - 기본 파인튜닝 스크립트

### 특성
- **목적**: 일반적인 Supervised Fine-Tuning (SFT)
- **대상**: 처음 파인튜닝을 시작하는 사용자
- **안정성**: 높음 (가장 안전한 기본 설정)
- **메모리 사용**: 보통
- **속도**: 보통
- **복잡성**: 낮음 (간단한 구조)

### 권장 사용 케이스
- 파인튜닝 처음 시도
- 안정성이 중요한 프로덕션 환경
- 작은 데이터셋 (1K~10K 샘플)
- 실험 및 프로토타이핑

### 권장 설정

#### 기본 설정 (8GB GPU)
```bash
python train.py \
  --model-name "EleutherAI/polyglot-ko-1.3b" \
  --dataset-name "maywell/korean_textbooks" \
  --output-dir "basic_finetuned_model" \
  --batch-size 2 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --max-length 512 \
  --lora-r 64 \
  --lora-alpha 128 \
  --use-8bit \
  --limit-samples 1000
```

#### 고성능 설정 (12GB+ GPU)
```bash
python train.py \
  --model-name "beomi/KoAlpaca-Polyglot-1.3B" \
  --dataset-name "maywell/korean_textbooks" \
  --output-dir "basic_finetuned_model" \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --max-length 1024 \
  --lora-r 128 \
  --lora-alpha 256 \
  --use-8bit \
  --merge-and-save
```

---

## 2. continue_train.py - 증분 학습 스크립트

### 특성
- **목적**: 기존 파인튜닝된 모델에서 추가 학습
- **대상**: 이미 파인튜닝된 모델을 보유한 사용자
- **안정성**: 높음
- **메모리 사용**: 보통
- **속도**: 빠름 (기존 지식 활용)
- **복잡성**: 중간 (모드 선택 필요)

### 권장 사용 케이스
- 기존 모델 성능 개선
- 새로운 도메인 데이터 추가 학습
- 점진적 데이터셋 확장
- 모델 버전 관리

### 권장 설정

#### 기존 어댑터 계속 학습
```bash
python continue_train.py \
  --base-model "EleutherAI/polyglot-ko-1.3b" \
  --finetuned-model "my_previous_model" \
  --dataset-name "maywell/korean_textbooks" \
  --output-dir "my_continued_model" \
  --mode continue \
  --batch-size 2 \
  --epochs 3 \
  --learning-rate 1e-4 \
  --max-length 1024 \
  --lora-r 64 \
  --lora-alpha 128 \
  --use-8bit
```

#### 병합 후 새 어댑터 학습
```bash
python continue_train.py \
  --base-model "EleutherAI/polyglot-ko-1.3b" \
  --finetuned-model "my_previous_model" \
  --dataset-name "maywell/korean_textbooks" \
  --output-dir "my_merged_continued_model" \
  --mode merge_and_new \
  --batch-size 2 \
  --epochs 5 \
  --learning-rate 5e-5 \
  --max-length 1024 \
  --lora-r 128 \
  --lora-alpha 256 \
  --use-8bit \
  --merge-and-save
```

---

## 3. optimized_train.py - 고성능 최적화 스크립트

### 특성
- **목적**: RTX 3060 12GB 최적화된 고성능 파인튜닝
- **대상**: 성능과 효율성을 중시하는 고급 사용자
- **안정성**: 중간 (고급 최적화 기능 사용)
- **메모리 사용**: 최적화됨 (가장 효율적)
- **속도**: 빠름 (최고 성능)
- **복잡성**: 높음 (많은 최적화 옵션)

### 권장 사용 케이스
- RTX 3060 12GB 사용자
- 대용량 데이터셋 (10K+ 샘플)
- 긴 시퀀스 처리 (2048+ 토큰)
- 프로덕션 레벨 모델 훈련
- 최고 성능이 필요한 경우

### 권장 설정

#### RTX 3060 12GB 최적화 설정
```bash
python optimized_train.py \
  --model-name "beomi/KoAlpaca-Polyglot-1.3B" \
  --dataset-name "maywell/korean_textbooks" \
  --output-dir "optimized_model" \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --lora-r 128 \
  --lora-alpha 256 \
  --lora-dropout 0.05 \
  --use-4bit \
  --use-flash-attention \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --lr-scheduler cosine \
  --merge-and-save
```

#### 메모리 절약 설정 (8GB GPU)
```bash
python optimized_train.py \
  --model-name "EleutherAI/polyglot-ko-1.3b" \
  --dataset-name "maywell/korean_textbooks" \
  --output-dir "optimized_model_8gb" \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --epochs 3 \
  --learning-rate 5e-5 \
  --max-length 1024 \
  --lora-r 64 \
  --lora-alpha 128 \
  --use-4bit \
  --warmup-ratio 0.05 \
  --weight-decay 0.005
```

#### 최대 성능 설정 (16GB+ GPU)
```bash
python optimized_train.py \
  --model-name "beomi/KoAlpaca-Polyglot-1.3B" \
  --dataset-name "maywell/korean_textbooks" \
  --output-dir "high_performance_model" \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --epochs 10 \
  --learning-rate 2e-4 \
  --max-length 4096 \
  --lora-r 256 \
  --lora-alpha 512 \
  --lora-dropout 0.1 \
  --use-flash-attention \
  --warmup-ratio 0.15 \
  --weight-decay 0.02 \
  --lr-scheduler cosine \
  --merge-and-save
```

---

## 스크립트 선택 가이드

### GPU 메모리별 권장사항

| GPU 메모리 | 권장 스크립트 | 설정 |
|-----------|-------------|------|
| 6GB 이하 | train.py | batch-size=1, max-length=256, use-8bit |
| 8GB | train.py 또는 optimized_train.py | batch-size=2-4, max-length=512-1024, use-4bit |
| 12GB | optimized_train.py | batch-size=8, max-length=2048, use-4bit |
| 16GB+ | optimized_train.py | batch-size=16+, max-length=4096 |

### 사용 목적별 권장사항

| 목적 | 권장 스크립트 | 이유 |
|-----|-------------|------|
| 처음 시도 | train.py | 안정성과 단순함 |
| 기존 모델 개선 | continue_train.py | 효율적인 증분 학습 |
| 고성능 필요 | optimized_train.py | 최적화된 성능 |
| 프로덕션 배포 | optimized_train.py + merge-and-save | 완전한 모델 생성 |
| 실험/연구 | train.py | 빠른 프로토타이핑 |

### 데이터셋 크기별 권장사항

| 데이터 크기 | 권장 스크립트 | 설정 |
|-----------|-------------|------|
| < 1K 샘플 | train.py | epochs=3-5, limit-samples 사용 안 함 |
| 1K-10K | train.py 또는 continue_train.py | epochs=3-5 |
| 10K-50K | optimized_train.py | epochs=3-5, gradient-accumulation-steps 조정 |
| 50K+ | optimized_train.py | epochs=1-3, 효율적인 배치 설정 |

---

## 공통 팁

### 메모리 부족 시 해결책
1. `--batch-size` 줄이기
2. `--max-length` 줄이기
3. `--use-4bit` 또는 `--use-8bit` 사용
4. `--gradient-accumulation-steps` 늘리기
5. `--limit-samples` 사용하여 데이터 제한

### 성능 향상 팁
1. `--use-flash-attention` 사용 (optimized_train.py)
2. 적절한 `--warmup-ratio` 설정
3. `--lr-scheduler cosine` 사용
4. `--gradient-accumulation-steps`로 효과적 배치 크기 조정

### 안정성 확보 팁
1. 작은 학습률로 시작 (1e-5 ~ 2e-4)
2. 적은 에포크로 시작 (3-5)
3. 정기적인 체크포인트 저장
4. 훈련 로그 모니터링