# 한국어 LLM TRL 파인튜닝 프로젝트 - 종합 가이드

## 앞으로 과제

1. **"/media/proidea/hdd/Downloads/030.웹데이터_기반_한국어_말뭉치_데이터"를 활용해서 학습용 jsonl을 만드는 코드를 추가하고 나서 실질적으로 추가 학습하기**
2. **현재는 text기반 학습데이터로만 진행하는데 다양한 포맷의 데이터포맷 1(instruction, output, url), 포맷2(title, subtitle, content, board, writer, write_date, url, source_site)등 으로 하는 코드를 작성**
3. **학습시 loss을 낮추기 위한 방안 모색**

---

## 📖 개요

이 프로젝트는 **Ubuntu 24.04**와 **NVIDIA RTX 3060 (12GB VRAM)** 환경에서 Hugging Face의 TRL(Transformer Reinforcement Learning)을 사용하여 한국어 대규모 언어 모델을 효과적으로 파인튜닝하는 방법을 제공합니다. 메모리 최적화 기법(8비트 양자화, LoRA)을 통해 고사양 장비 없이도 LLM 파인튜닝이 가능합니다.

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론 후 디렉터리 이동
cd UseTRLFinetune

# 설정 스크립트 실행 권한 부여
chmod +x setup.sh

# 자동 환경 설정
./setup.sh

# 가상환경 활성화
source korean-llm-env/bin/activate
```

### 2. 기본 파인튜닝 실행
```bash
# 기본 TRL PPO 기반 파인튜닝
python train.py

# 또는 간단한 Trainer 기반 파인튜닝
python simple_train.py --data-path sample_korean_data.jsonl
```

### 3. 모델 테스트
```bash
# 기본 테스트
python test.py

# 빠른 효과 확인
python quick_test.py

# 대화형 테스트
python inference.py --interactive
```

## 📁 프로젝트 구조

```
UseTRLFinetune/
├── README.md                      # 이 문서
├── requirements.txt               # Python 의존성
├── setup.sh                       # 환경 설정 스크립트
├── sample_korean_data.jsonl       # 샘플 학습 데이터
├── 
├── ===== 🎯 핵심 훈련 스크립트 =====
├── train.py                       # TRL PPO 기반 파인튜닝 (기본)
├── simple_train.py                # Hugging Face Trainer 기반 (안정적)
├── optimized_train.py             # RTX 3060 최적화 (권장)
├── custom_train.py                # 커스텀 데이터셋 파인튜닝
├── continue_train.py              # 기존 모델 추가 학습
├── 
├── ===== 📊 평가 및 추론 =====
├── evaluate.py                    # 고급 평가 도구
├── evaluate_simple.py             # 간단한 키워드 기반 평가
├── inference.py                   # 기본 추론 스크립트
├── inference_safe.py              # 안정적인 추론
├── chat_test.py                   # 콘솔 채팅 테스트
├── test.py                        # 기본 테스트
├── quick_test.py                  # 빠른 효과 확인
├── 
├── ===== 🗂️ 데이터 수집 및 처리 =====
├── main.py                        # 메인 조율 스크립트
├── data_processor.py              # 데이터 가공 및 변환
├── demo.py                        # 데이터 수집 데모
├── data_collectors/
│   ├── web_scraper.py            # 웹 스크래핑
│   ├── pdf_extractor.py          # PDF 텍스트 추출
│   └── supabase_manager.py       # Supabase 연동
├── 
├── ===== 🔧 분석 및 유틸리티 =====
├── analyze_model.py               # 모델 구조 분석
├── load.py                        # 기본 모델 로딩 예제
├── check_ppo_params.py            # PPO 파라미터 확인
├── test_ppo_trainer.py            # PPO Trainer 테스트
├── 
├── ===== 📂 생성된 모델 및 데이터 =====
├── korean-llm-env/                # Python 가상환경
├── my_korean_finetuned_model/     # 기본 파인튜닝 모델
├── my_optimized_model/            # 최적화 파인튜닝 모델
├── my_continued_model/            # 추가 학습 모델
├── web_data.jsonl                 # 웹 스크래핑 데이터
└── evaluation_results.json        # 평가 결과
```

## 🎯 핵심 훈련 스크립트

### 1. `train.py` - TRL PPO 기반 파인튜닝 (기본)
**기능**: TRL의 PPO(Proximal Policy Optimization)를 사용한 강화학습 기반 파인튜닝
**데이터**: NSMC(네이버 영화 리뷰) 데이터셋
**목표**: 긍정적인 한국어 텍스트 생성 학습

```bash
python train.py
```

**특징**:
- NSMC 데이터셋 자동 로드
- 한국어 감정 분석 모델을 보상 함수로 사용
- 8비트 양자화 + LoRA 최적화
- 소규모 데이터로 빠른 테스트

### 2. `simple_train.py` - Hugging Face Trainer 기반 (안정적)
**기능**: PPO 대신 Hugging Face Trainer를 사용한 안정적인 파인튜닝
**권장**: API 호환성 문제가 있을 때 사용

```bash
python simple_train.py --data-path your_data.jsonl \
                       --output-dir my_simple_model \
                       --epochs 3 \
                       --batch-size 4
```

**특징**:
- PPO 대신 표준 언어 모델링 학습
- 더 안정적이고 예측 가능한 결과
- 커스텀 데이터셋 지원

### 3. `optimized_train.py` - RTX 3060 최적화 (⭐ 권장)
**기능**: RTX 3060 12GB 환경에 최적화된 파인튜닝
**권장**: 가장 효율적이고 고성능

```bash
python optimized_train.py --data-path your_data.jsonl \
                         --output-dir my_optimized_model \
                         --batch-size 8 \
                         --max-length 2048 \
                         --lora-r 128 \
                         --use-4bit  # 4비트 양자화 (선택)
```

**특징**:
- RTX 3060 12GB에 최적화된 하이퍼파라미터
- 4비트/8비트 양자화 선택 가능
- Flash Attention 지원
- 고급 LoRA 설정 (rank 128, alpha 256)
- 동적 패딩 및 그래디언트 체크포인팅

### 4. `custom_train.py` - 커스텀 데이터셋 파인튜닝
**기능**: 사용자 제공 JSONL 데이터로 파인튜닝

```bash
python custom_train.py --data-path custom_data.jsonl \
                       --output-dir my_custom_model \
                       --use-reward-model \
                       --epochs 5 \
                       --limit-samples 1000
```

**특징**:
- 임의의 JSONL 데이터 지원
- 보상 모델 사용/미사용 선택 가능
- 메모리 절약을 위한 샘플 수 제한

### 5. `continue_train.py` - 기존 모델 추가 학습
**기능**: 이미 파인튜닝된 모델에서 추가 학습

```bash
# 기존 어댑터에서 계속 학습
python continue_train.py --finetuned-model my_korean_finetuned_model \
                        --data-path new_data.jsonl \
                        --output-dir my_continued_model \
                        --mode continue

# 어댑터를 모델과 병합 후 새 어댑터 학습
python continue_train.py --finetuned-model my_korean_finetuned_model \
                        --data-path new_data.jsonl \
                        --output-dir my_continued_model \
                        --mode merge_and_new \
                        --merge-and-save
```

**특징**:
- 두 가지 학습 방식 지원
- 어댑터 병합 옵션
- 다양한 양자화 옵션

## 📊 평가 및 추론 스크립트

### 1. `evaluate.py` - 고급 평가 도구 (⭐ 권장)
**기능**: 베이스 모델과 파인튜닝된 모델의 포괄적 비교

```bash
python evaluate.py
```

**특징**:
- 감정 분석 기반 성능 측정
- 여러 프롬프트로 모델 비교
- 테스트 데이터셋 평가
- JSON 결과 저장

### 2. `evaluate_simple.py` - 간단한 키워드 기반 평가
**기능**: 외부 모델 없이 키워드로 간단 평가

```bash
python evaluate_simple.py
```

**특징**:
- 긍정/부정 키워드 기반 평가
- 빠른 실행
- 외부 의존성 최소화

### 3. `inference.py` - 기본 추론 스크립트
**기능**: 파인튜닝된 모델로 텍스트 생성

```bash
# 대화형 모드
python inference.py --interactive

# 단일 프롬프트
python inference.py --prompt "이 영화는 정말" --max_tokens 100

# 기본 테스트
python inference.py
```

**특징**:
- 대화형 모드 지원
- 다양한 생성 옵션
- 모델 경로 자동 감지

### 4. `chat_test.py` - 콘솔 채팅 테스트
**기능**: 파인튜닝된 모델과 실시간 대화

```bash
python chat_test.py
```

**특징**:
- 사용 가능한 모델 자동 탐지
- 대화형 설정
- 실시간 생성 통계
- 설정 확인 기능

### 5. `quick_test.py` - 빠른 효과 확인
**기능**: 파인튜닝 효과를 빠르게 확인

```bash
python quick_test.py
```

**특징**:
- 베이스 vs 파인튜닝 모델 직접 비교
- 긍정 키워드 카운트
- 빠른 실행

## 🗂️ 데이터 수집 및 처리

### 1. `main.py` - 메인 조율 스크립트
**기능**: 데이터 수집부터 모델 훈련까지 통합 관리

```bash
# 웹 스크래핑
python main.py --collect --web "https://example.com" "https://example2.com" \
               --output web_data.jsonl

# 1DEPTH 웹 크롤링
python main.py --collect --web "https://example.com" --web-crawl \
               --max-pages 50 --crawl-delay 2

# PDF 텍스트 추출
python main.py --collect --pdf "https://example.com/paper.pdf" \
               --output pdf_data.jsonl

# 텍스트 파일 처리
python main.py --collect --text file1.txt file2.txt \
               --output text_data.jsonl

# Supabase 연동
python main.py --collect --supabase \
               --supabase-url "your-url" --supabase-key "your-key"

# 수집 + 훈련 통합
python main.py --collect --web "https://example.com" \
               --train --output-dir my_trained_model
```

### 2. `data_processor.py` - 데이터 가공 및 변환
**기능**: 다양한 형태의 데이터를 학습용으로 변환

```python
from data_processor import (
    create_dataset_from_texts,
    create_dataset_from_jsonl,
    save_texts_to_jsonl
)

# 텍스트 리스트를 데이터셋으로 변환
texts = ["텍스트1", "텍스트2", "텍스트3"]
dataset = create_dataset_from_texts(texts)

# JSONL 파일을 데이터셋으로 변환
dataset = create_dataset_from_jsonl("data.jsonl")

# 텍스트를 JSONL로 저장
save_texts_to_jsonl(texts, "output.jsonl", append_mode=True)
```

### 3. 데이터 수집기 (`data_collectors/`)

#### `web_scraper.py` - 웹 스크래핑
```python
from data_collectors.web_scraper import extract_text_from_url, crawl_website_depth_1

# 단일 URL 처리
result = extract_text_from_url("https://example.com")

# 1DEPTH 크롤링
results = crawl_website_depth_1("https://example.com", max_pages=20)
```

#### `pdf_extractor.py` - PDF 텍스트 추출
```python
from data_collectors.pdf_extractor import extract_text_from_pdf_url

# PDF URL에서 텍스트 추출
result = extract_text_from_pdf_url("https://example.com/paper.pdf")
```

#### `supabase_manager.py` - Supabase 연동
```python
from data_collectors.supabase_manager import SupabaseManager

# Supabase 연동
manager = SupabaseManager(url="your-url", key="your-key")
manager.create_training_data_table()
manager.insert_web_data(web_result)
```

### 4. `demo.py` - 종합 데모
**기능**: 전체 파이프라인 데모

```bash
# 웹 스크래핑 데모
python demo.py --demo-mode web

# PDF 추출 데모  
python demo.py --demo-mode pdf

# 전체 데모 (수집 + 훈련)
python demo.py --demo-mode all

# 데이터 수집만 (훈련 제외)
python demo.py --demo-mode all --skip-training
```

## 🔧 분석 및 유틸리티 스크립트

### 1. `analyze_model.py` - 모델 구조 분석
**기능**: polyglot-ko 모델의 구조를 분석하여 LoRA 타겟 모듈 찾기

```bash
python analyze_model.py
```

### 2. `load.py` - 기본 모델 로딩 예제
**기능**: 8비트 양자화와 LoRA 설정 예제

### 3. `check_ppo_params.py` / `test_ppo_trainer.py` 
**기능**: TRL PPOTrainer의 API 확인 및 테스트

## ⚙️ 주요 설정 및 최적화

### 1. 메모리 최적화 기법

#### 8비트 양자화
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

#### 4비트 양자화 (더 많은 메모리 절약)
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

#### LoRA 설정
```python
from peft import LoraConfig

# 기본 설정
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 고성능 설정 (optimized_train.py)
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 2. RTX 3060 12GB 최적화 권장사항

| 설정 | 기본값 | RTX 3060 최적화 |
|------|--------|-----------------|
| batch_size | 4 | 8 |
| max_length | 512 | 2048 |
| lora_r | 64 | 128 |
| lora_alpha | 16 | 256 |
| gradient_accumulation_steps | 2 | 4 |
| 양자화 | 8bit | 4bit (선택) |

### 3. 주요 하이퍼파라미터

```python
# 학습률
learning_rate = 1e-4  # 안정적
learning_rate = 2e-4  # 빠른 학습

# 배치 크기
batch_size = 4   # 기본
batch_size = 8   # RTX 3060 최적화

# 에포크 수
epochs = 3  # 빠른 테스트
epochs = 5  # 일반적
epochs = 10 # 고품질 (오버피팅 주의)

# 시퀀스 길이
max_length = 512   # 기본
max_length = 1024  # 긴 문맥
max_length = 2048  # 매우 긴 문맥 (메모리 주의)
```

## 🚨 문제 해결

### 1. CUDA 메모리 부족 (OOM)
```bash
# 메모리 사용량 줄이기
python optimized_train.py --data-path data.jsonl \
                         --batch-size 4 \
                         --max-length 1024 \
                         --use-4bit
```

**해결 방법**:
- `--batch-size` 줄이기 (8 → 4 → 2)
- `--max-length` 줄이기 (2048 → 1024 → 512)
- `--use-4bit` 옵션 사용
- `--gradient-accumulation-steps` 늘리기

### 2. PPOTrainer API 오류
```bash
# 안정적인 대안 사용
python simple_train.py --data-path data.jsonl
```

### 3. 모델 로딩 실패
```bash
# 안전한 추론 스크립트 사용
python inference_safe.py --interactive
```

### 4. 의존성 설치 오류
```bash
# Python 개발 헤더 설치
sudo apt install python3.12-dev

# bitsandbytes 재설치
pip uninstall bitsandbytes
pip install bitsandbytes
```

### 5. 성능 최적화

#### 학습 속도 향상
- Flash Attention 사용: `--use-flash-attention`
- 그래디언트 체크포인팅: 자동 적용
- 데이터로더 멀티프로세싱: `dataloader_num_workers=4`

#### 메모리 사용량 감소
- 4비트 양자화: `--use-4bit`
- 작은 배치 크기: `--batch-size 2`
- 짧은 시퀀스: `--max-length 512`

#### 학습 품질 향상
- 높은 LoRA rank: `--lora-r 128`
- 더 많은 에포크: `--epochs 10`
- 큰 데이터셋: `--limit-samples` 제거

## 📈 모델 성능 개선 방법

### 1. 데이터 품질 향상
- 더 많은 고품질 한국어 데이터 수집
- 데이터 전처리 및 필터링 강화
- 도메인 특화 데이터 추가

### 2. 하이퍼파라미터 튜닝
- 학습률 조정 (1e-5 ~ 5e-4)
- LoRA rank 증가 (64 → 128 → 256)
- 배치 크기 최적화

### 3. 고급 기법 적용
- Flash Attention 2 사용
- Gradient Checkpointing
- Mixed Precision Training (BF16)

## 🎯 사용 시나리오별 가이드

### 1. 빠른 테스트 (5분)
```bash
python train.py                    # 기본 데이터로 빠른 학습
python quick_test.py              # 효과 확인
```

### 2. 고품질 파인튜닝 (1-2시간)
```bash
python optimized_train.py --data-path large_data.jsonl \
                         --epochs 5 \
                         --batch-size 8 \
                         --max-length 2048
python evaluate.py                # 상세 평가
```

### 3. 커스텀 데이터 학습
```bash
# 1. 데이터 수집
python main.py --collect --web "https://your-site.com" \
               --web-crawl --max-pages 100

# 2. 학습
python custom_train.py --data-path web_data.jsonl \
                      --epochs 3

# 3. 평가
python evaluate.py
```

### 4. 추가 학습
```bash
# 기존 모델에서 새 데이터로 추가 학습
python continue_train.py --finetuned-model my_optimized_model \
                        --data-path new_data.jsonl \
                        --mode continue
```

## 📄 라이선스 및 참고자료

### 참고자료
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [PEFT](https://github.com/huggingface/peft)
- [EleutherAI Polyglot-Ko](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
- [NSMC Dataset](https://huggingface.co/datasets/nsmc)

### 시스템 요구사항
- **OS**: Ubuntu 24.04 (다른 Linux 배포판도 가능)
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) 또는 유사한 성능
- **RAM**: 16GB 이상 권장
- **저장공간**: 20GB 이상
- **Python**: 3.12+

### 주요 의존성
- PyTorch 2.0+
- Transformers 4.30+
- TRL 0.7+
- PEFT 0.4+
- BitsAndBytes 0.41+
- Datasets 2.0+

이 프로젝트는 교육 및 연구 목적으로 제공됩니다. 각 스크립트의 세부 옵션은 `python script_name.py --help`로 확인할 수 있습니다.