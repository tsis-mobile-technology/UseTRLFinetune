# 한국어 LLM TRL 파인튜닝 프로젝트 - 완전 통합 가이드

## 🆕 최신 업데이트 (2025년 6월 29일)

### ✅ 실행/테스트/환경/디버깅 관련 스크립트가 다음과 같이 통합되었습니다:

- **run_scripts.py** : 모든 실행/테스트/가이드 스크립트 통합
    - CPU/최적화/참조모델/테스트 등 다양한 실행 모드 지원
    - 사용법: `python run_scripts.py list` 로 명령어 목록 확인
- **fix_environment.py** : GPU/CUDA 환경 진단 및 복구 통합
    - 드라이버/패키지/환경변수/설치 자동화
    - 사용법: `python fix_environment.py list` 로 명령어 목록 확인
- **debug_tools.py** : PPO/참조모델/메모리/패키지 등 디버깅 도구 통합
    - 사용법: `python debug_tools.py list` 로 명령어 목록 확인

> 기존 run_, fix_, debug_로 시작하는 개별 파일들과 분산된 .md 문서들은 모두 `archive_old_tests/` 폴더로 이동되었습니다.

---

## 📖 개요

이 프로젝트는 **Ubuntu 24.04**와 **NVIDIA RTX 3060 (12GB VRAM)** 환경에서 Hugging Face의 TRL(Transformer Reinforcement Learning)을 사용하여 한국어 대규모 언어 모델을 효과적으로 파인튜닝하는 종합적인 시스템입니다. 메모리 최적화 기법(8비트 양자화, LoRA)을 통해 고사양 장비 없이도 LLM 파인튜닝이 가능하며, 다양한 데이터 소스에서 수집부터 파인튜닝까지 전체 파이프라인을 제공합니다.

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

### 2. 환경 검증 및 문제 해결
```bash
# 종합 환경 검증
python environment_validation.py

# GPU/CUDA 문제가 있는 경우 환경 수정
python fix_environment.py full-fix

# 디버깅이 필요한 경우
python debug_tools.py all
```

### 3. 기본 파인튜닝 실행
```bash
# 기본 SFT 파인튜닝
python train.py --dataset-name maywell/korean_textbooks --use-8bit

# 또는 통합 실행 도구 사용
python run_scripts.py cpu-train
```

### 4. 모델 테스트 및 평가
```bash
# 대화형 테스트
python inference.py --interactive

# 종합 프로그램 테스트
python program_testing.py
```

## 📁 프로젝트 구조

```
UseTRLFinetune/
├── README.md                      # 이 통합 문서
├── requirements.txt               # Python 의존성
├── setup.sh                       # 환경 설정 스크립트
├── sample_korean_data.jsonl       # 샘플 학습 데이터
├── 
├── ===== 🎯 핵심 훈련 스크립트 =====
├── train.py                       # 기본 SFT 파인튜닝 (안정적)
├── simple_train.py                # Hugging Face Trainer 기반
├── optimized_train.py             # RTX 3060 최적화 (권장)
├── custom_train.py                # 커스텀 데이터셋 파인튜닝
├── continue_train.py              # 기존 모델 추가 학습
├── train_fixed.py                 # PPO 오류 완전 해결 버전
├── 
├── ===== 🔧 통합 실행/환경/디버깅 도구 =====
├── run_scripts.py                 # 실행 스크립트 통합 도구
├── fix_environment.py             # 환경 수정 통합 도구
├── debug_tools.py                 # 디버깅 통합 도구
├── 
├── ===== 📊 평가 및 추론 =====
├── evaluate.py                    # 고급 평가 도구
├── evaluate_simple.py             # 간단한 키워드 기반 평가
├── inference.py                   # 기본 추론 스크립트
├── inference_safe.py              # 안정적인 추론
├── chat_test.py                   # 콘솔 채팅 테스트
├── 
├── ===== 🧪 테스트 및 검증 =====
├── environment_validation.py      # 환경 설정 및 검증 (통합)
├── program_testing.py             # 주요 프로그램 테스트 (통합)
├── archive_old_tests/             # 기존 개별 테스트 파일들
├── 
├── ===== 🗂️ 데이터 수집 및 처리 =====
├── main.py                        # 메인 조율 스크립트
├── data_processor.py              # 데이터 가공 및 변환
├── demo.py                        # 데이터 수집 데모
├── data_collectors/
│   ├── web_scraper.py            # 웹 스크래핑
│   ├── pdf_extractor.py          # PDF 텍스트 추출
│   ├── reddit_collector.py       # Reddit 데이터 수집
│   └── supabase_manager.py       # Supabase 연동
├── 
├── ===== 🔧 분석 및 유틸리티 =====
├── analyze_model.py               # 모델 구조 분석
├── load.py                        # 기본 모델 로딩 예제
├── 
├── ===== 📂 생성된 모델 및 데이터 =====
├── korean-llm-env/                # Python 가상환경
├── my_korean_finetuned_model/     # 기본 파인튜닝 모델
├── my_optimized_model/            # 최적화 파인튜닝 모델
├── my_continued_model/            # 추가 학습 모델
├── web_data.jsonl                 # 웹 스크래핑 데이터
└── evaluation_results.json        # 평가 결과
```

## 🎯 핵심 훈련 스크립트 상세 비교

### 1. `train.py` - 기본 SFT 파인튜닝 (⭐ 안정성 최우선)
**특성**:
- **목적**: 일반적인 Supervised Fine-Tuning (SFT)
- **대상**: 처음 파인튜닝을 시작하는 사용자
- **안정성**: 매우 높음 (가장 안전한 설정)
- **메모리 사용**: 보통
- **복잡성**: 낮음 (간단한 구조)

**권장 사용 케이스**:
- 파인튜닝 처음 시도
- 안정성이 중요한 프로덕션 환경
- 작은 데이터셋 (1K~10K 샘플)
- 실험 및 프로토타이핑

```bash
# 기본 설정 (8GB GPU)
python train.py \
  --dataset-name maywell/korean_textbooks \
  --output-dir my_finetuned_model \
  --batch-size 2 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --max-length 512 \
  --lora-r 64 \
  --lora-alpha 128 \
  --use-8bit \
  --limit-samples 1000

# 고성능 설정 (12GB+ GPU)
python train.py \
  --dataset-name maywell/korean_textbooks \
  --output-dir my_finetuned_model \
  --batch-size 4 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --max-length 1024 \
  --lora-r 128 \
  --lora-alpha 256 \
  --use-8bit \
  --merge-and-save

# 로컬 JSONL 데이터 사용
python train.py \
  --data-path custom_data.jsonl \
  --batch-size 2 \
  --max-length 1024 \
  --merge-and-save
```

### 2. `train_fixed.py` - PPO 오류 완전 해결 버전 (⭐ PPO 전용)
**특성**:
- **목적**: TRL PPO의 'tuple' object has no attribute 'logits' 오류 완전 해결
- **대상**: PPO 강화학습 파인튜닝이 필요한 경우
- **기술적 혁신**: FixedPPOTrainer 클래스와 동적 패치 시스템
- **해결된 문제들**: tuple 오류, CUDA OOM, 참조 모델 처리 문제

**핵심 개선사항**:
1. **FixedPPOTrainer 클래스**: PPOTrainer를 상속받아 참조 모델 문제 완전 해결
2. **동적 패치 시스템**: 런타임에 참조 모델의 출력 형식 자동 수정
3. **메모리 최적화**: 8-bit 양자화, gradient checkpointing, 동적 메모리 관리

```bash
# 권장 설정 (8-bit 양자화)
python train_fixed.py \
  --model_name EleutherAI/polyglot-ko-1.3b \
  --dataset_name maywell/korean_textbooks \
  --output_dir my_ppo_fixed_model \
  --batch_size 2 \
  --mini_batch_size 1 \
  --use_8bit_quantization \
  --enable_gradient_checkpointing

# 최소 메모리 모드 (6-8GB GPU)
python train_fixed.py \
  --batch_size 1 \
  --input_max_text_length 128 \
  --max_new_tokens 8 \
  --dataset_sample_size 10 \
  --lora_r 8 \
  --use_8bit_quantization

# 고성능 모드 (12GB+ GPU)
python train_fixed.py \
  --model_name EleutherAI/polyglot-ko-1.3b \
  --dataset_name maywell/korean_textbooks \
  --output_dir my_ppo_high_perf_model \
  --batch_size 2 \
  --mini_batch_size 1 \
  --input_max_text_length 256 \
  --max_new_tokens 16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --enable_gradient_checkpointing
```

### 3. `optimized_train.py` - RTX 3060 최적화 (⭐ 최고 성능)
**특성**:
- **목적**: RTX 3060 12GB 최적화된 고성능 파인튜닝
- **대상**: 성능과 효율성을 중시하는 고급 사용자
- **메모리 사용**: 최적화됨 (가장 효율적)
- **속도**: 빠름 (최고 성능)
- **복잡성**: 높음 (많은 최적화 옵션)

**권장 사용 케이스**:
- RTX 3060 12GB 사용자
- 대용량 데이터셋 (10K+ 샘플)
- 긴 시퀀스 처리 (2048+ 토큰)
- 프로덕션 레벨 모델 훈련

```bash
# RTX 3060 12GB 최적화 설정
python optimized_train.py \
  --data-path your_data.jsonl \
  --output-dir my_optimized_model \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --lora-r 128 \
  --lora-alpha 256 \
  --use-4bit \
  --use-flash-attention \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --lr-scheduler cosine \
  --merge-and-save

# 메모리 절약 설정 (8GB GPU)
python optimized_train.py \
  --data-path your_data.jsonl \
  --output-dir optimized_model_8gb \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --epochs 3 \
  --learning-rate 5e-5 \
  --max-length 1024 \
  --lora-r 64 \
  --lora-alpha 128 \
  --use-4bit
```

### 4. `continue_train.py` - 기존 모델 추가 학습 (⭐ 점진적 개선)
**특성**:
- **목적**: 기존 파인튜닝된 모델에서 추가 학습
- **대상**: 이미 파인튜닝된 모델을 보유한 사용자
- **안정성**: 높음
- **속도**: 빠름 (기존 지식 활용)
- **복잡성**: 중간 (모드 선택 필요)

**지원하는 학습 방식**:
1. **기존 어댑터 계속 학습 (Continue Training)**: 가장 효율적이고 권장되는 방법
2. **병합 후 새 어댑터 학습 (Merge and New Training)**: 기존 학습과 새 학습을 명확히 분리

**권장 사용 케이스**:
- 기존 모델 성능 개선
- 새로운 도메인 데이터 추가 학습
- 점진적 데이터셋 확장
- 모델 버전 관리

```bash
# 기존 어댑터에서 계속 학습
python continue_train.py \
  --base-model EleutherAI/polyglot-ko-1.3b \
  --finetuned-model my_korean_finetuned_model \
  --data-path new_data.jsonl \
  --output-dir my_continued_model \
  --mode continue \
  --batch-size 4 \
  --epochs 3 \
  --use-8bit

# 어댑터 병합 후 새 어댑터 학습
python continue_train.py \
  --base-model EleutherAI/polyglot-ko-1.3b \
  --finetuned-model my_korean_finetuned_model \
  --data-path new_data.jsonl \
  --output-dir my_merged_new_model \
  --mode merge_and_new \
  --batch-size 2 \
  --epochs 5 \
  --learning-rate 5e-5 \
  --use-8bit \
  --merge-and-save

# 학습 후 병합된 전체 모델 저장
python continue_train.py \
  --finetuned-model my_optimized_model \
  --data-path new_training_data.jsonl \
  --output-dir my_continued_model \
  --mode continue \
  --merge-and-save \
  --batch-size 4 \
  --epochs 3 \
  --use-8bit
```

### 5. `custom_train.py` - 커스텀 데이터셋 파인튜닝
**기능**: 사용자 제공 JSONL 데이터로 파인튜닝

```bash
python custom_train.py \
  --data-path custom_data.jsonl \
  --use-reward-model \
  --epochs 5 \
  --limit-samples 1000
```

## 🔧 통합 도구 사용법

### 1. 실행 스크립트 통합 도구 (`run_scripts.py`)
모든 실행 관련 스크립트를 하나로 통합

```bash
# 사용 가능한 명령어 확인
python run_scripts.py list

# CPU 모드 훈련
python run_scripts.py cpu-train

# 참조 모델 테스트
python run_scripts.py test-ref-create

# 수정된 훈련 가이드 보기
python run_scripts.py guide-fixed
```

### 2. 환경 수정 통합 도구 (`fix_environment.py`)
GPU/CUDA 환경 진단 및 자동 복구

```bash
# 사용 가능한 명령어 확인
python fix_environment.py list

# 전체 환경 복구 (추천)
python fix_environment.py full-fix

# 시스템 진단만
python fix_environment.py diagnose

# PyTorch GPU 버전 설치
python fix_environment.py install-pytorch
```

### 3. 디버깅 통합 도구 (`debug_tools.py`)
모든 디버깅 기능을 하나로 통합

```bash
# 사용 가능한 명령어 확인
python debug_tools.py list

# 모든 디버깅 테스트 실행
python debug_tools.py all

# PPO 메소드 확인
python debug_tools.py ppo-methods

# 참조 모델 디버깅
python debug_tools.py ref-model

# 메모리 사용량 분석
python debug_tools.py memory
```

## 🗂️ 완전한 맞춤형 데이터 수집 및 처리 시스템

### 1. 메인 조율 스크립트 (`main.py`)
데이터 수집부터 모델 훈련까지 통합 관리

**주요 기능**:
- 다양한 소스(웹페이지, PDF, 텍스트 파일)에서 텍스트 데이터 수집
- Supabase를 이용한 데이터 저장 및 관리 기능
- 수집된 데이터로 한국어 언어 모델 효과적 파인튜닝

```bash
# 웹페이지에서 데이터 수집 (단일 페이지)
python main.py --collect --web "https://ko.wikipedia.org/wiki/인공지능" "https://ko.wikipedia.org/wiki/기계학습" --output "web_data.jsonl"

# 1DEPTH 웹 크롤링 (메인 페이지 + 링크된 하위 페이지들)
python main.py --collect --web "https://news.hada.io/show" --web-crawl --max-pages 20 --crawl-delay 1 --output "crawl_data.jsonl"

# PDF에서 데이터 수집
python main.py --collect --pdf "https://example.com/sample.pdf" --output "pdf_data.jsonl"

# 텍스트 파일에서 데이터 수집
python main.py --collect --text "sample1.txt" "sample2.txt" --output "text_data.jsonl"

# 여러 소스를 결합하여 수집
python main.py --collect --web "https://ko.wikipedia.org/wiki/인공지능" --pdf "https://example.com/sample.pdf" --text "sample.txt" --output "combined_data.jsonl"

# 수집 + 훈련 통합
python main.py --collect --web "https://example.com" \
               --train --output-dir my_trained_model

# Supabase 통합 사용
python main.py --collect --web "https://ko.wikipedia.org/wiki/인공지능" --supabase --supabase-url "your-url" --supabase-key "your-key"

# Supabase에서 데이터 내보내기
python main.py --export-supabase "exported_data.jsonl" --supabase-url "your-url" --supabase-key "your-key"
```

#### 1DEPTH 웹 크롤링 기능
`--web-crawl` 옵션을 사용하면 지정된 URL에서 링크된 하위 페이지들을 자동으로 찾아서 텍스트를 추출합니다:

- `--max-pages`: 크롤링할 최대 페이지 수 (기본값: 20)
- `--crawl-delay`: 페이지 간 지연 시간 초 단위 (기본값: 1)

```bash
# news.hada.io 메인 페이지와 링크된 20개 하위 페이지에서 텍스트 추출
python main.py --collect --web "https://news.hada.io/show" --web-crawl --max-pages 20 --crawl-delay 2 --output "hada_news_data.jsonl"
```

### 2. 지원되는 데이터 형식

#### 기본 text 형식 (현재 지원)
```json
{"text": "여기에 훈련 텍스트가 들어갑니다."}
```

#### instruction-output 형식 (향후 지원 예정)
```json
{"instruction": "질문 내용", "output": "답변 내용", "url": "출처 URL"}
```

#### 메타데이터 포함 형식 (향후 지원 예정)
```json
{
  "title": "제목",
  "subtitle": "부제목", 
  "content": "본문 내용",
  "board": "게시판",
  "writer": "작성자",
  "write_date": "작성일",
  "url": "URL",
  "source_site": "출처 사이트"
}
```

### 3. Supabase 설정 (선택 사항)
Supabase를 사용하려면 다음 환경 변수를 설정합니다:

```bash
export SUPABASE_URL="your-supabase-url"
export SUPABASE_KEY="your-supabase-api-key"
```

### 4. 데모 실행
다양한 기능을 한 번에 시연하는 데모 스크립트:

```bash
# 모든 기능 데모
python demo.py --demo-mode all --output-dir "demo_output"

# 웹 스크래핑만 데모
python demo.py --demo-mode web --output-dir "demo_output"

# 데이터 수집만 하고 훈련은 건너뛰기
python demo.py --demo-mode all --output-dir "demo_output" --skip-training
```

## 📊 평가 및 추론 시스템

### 1. `evaluate.py` - 고급 평가 도구 (⭐ 권장)
베이스 모델과 파인튜닝된 모델의 포괄적 비교

```bash
python evaluate.py
```

**주요 기능**:
- 베이스 모델 vs 파인튜닝 모델 비교
- 감정 분석 기반 성능 측정
- 테스트 데이터셋 평가
- 결과 JSON 저장

### 2. `inference.py` - 추론 도구
파인튜닝된 모델로 텍스트 생성

```bash
# 대화형 모드
python inference.py --interactive

# 단일 프롬프트
python inference.py --prompt "이 영화는 정말" --max_tokens 100

# 배치 테스트
python inference.py
```

### 3. `chat_test.py` - 실시간 채팅 테스트
파인튜닝된 모델과 실시간 대화

```bash
python chat_test.py
```

## 🧪 테스트 및 검증 시스템

### 1. `environment_validation.py` - 환경 검증 (⭐ 통합)
프로젝트 환경이 올바르게 설정되었는지 종합 검증

```bash
python environment_validation.py
```

**주요 기능**:
- 시스템 정보 확인 (OS, 커널, Python 버전)
- NVIDIA 드라이버 및 CUDA 환경 검증
- PyTorch, Transformers, TRL, bitsandbytes 라이브러리 확인
- 데이터셋 로딩 테스트
- 모델 return_dict 동작 검증
- argparse 기능 테스트

### 2. `program_testing.py` - 프로그램 테스트 (⭐ 통합)
프로젝트의 핵심 기능들이 정상 작동하는지 테스트

```bash
python program_testing.py
```

**주요 기능**:
- create_reference_model 함수 테스트
- return_dict 설정을 통한 tuple 문제 해결 테스트
- 참조 모델 래퍼 솔루션 테스트
- PPOTrainer 생성 및 동작 테스트
- 메모리 최적화 설정 테스트
- 파인튜닝 효과 검증
- train.py 스크립트 실행 테스트

### 3. 테스트 파일 통합 완료
**개선사항**: 기존의 28개 개별 테스트 파일들을 2개의 통합된 파일로 재구성

**아카이브된 파일들** (`archive_old_tests/`):
- `check_*.py` (6개): 환경, GPU, TRL 등 검증
- `test_*.py` (18개): PPO, 참조 모델, 래퍼, 메모리 최적화 등
- `quick_test*.py` (3개): 빠른 테스트 스크립트
- 기존 run_, fix_, debug_ 파일들
- 기존 분산된 .md 문서들

## ⚙️ 상세 메모리 최적화 가이드

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

#### 주요 메모리 최적화 설정
```bash
# 기본 메모리 절약 설정
--batch_size 2                    # 4 → 2 (50% 감소)
--mini_batch_size 1               # 2 → 1 (50% 감소)
--input_max_text_length 256       # 2048 → 256 (87% 감소)
--max_new_tokens 16              # 32 → 16 (50% 감소)
--dataset_sample_size 50         # 100 → 50 (50% 감소)

# LoRA 파라미터 최적화
--lora_r 16                      # 128 → 16 (87% 감소)
--lora_alpha 32                  # 256 → 32 (87% 감소)

# 메모리 최적화 기법
--enable_gradient_checkpointing   # Gradient Checkpointing
--use_8bit_quantization          # 8-bit 양자화
--max_memory_per_gpu 10GB        # GPU 메모리 제한
```

### 2. GPU 메모리별 상세 권장 설정

| GPU 메모리 | 권장 스크립트 | 배치 크기 | 시퀀스 길이 | 양자화 | LoRA r | 예상 성능 |
|-----------|-------------|----------|------------|--------|---------|----------|
| 6GB 이하 | train.py | 1 | 256 | 8-bit | 16-32 | 낮음 |
| 8GB | train.py | 2-4 | 512-1024 | 4-bit | 32-64 | 중간 |
| 12GB | optimized_train.py | 8 | 2048 | 4-bit | 64-128 | 높음 |
| 16GB+ | optimized_train.py | 16+ | 4096 | 선택 | 128-256 | 최고 |

### 3. 메모리 사용량 추정표

| 설정 모드 | 예상 GPU 메모리 | 적합한 GPU | 처리량 |
|----------|----------------|------------|--------|
| 극한 절약 | ~2-4GB | RTX 3060, GTX 1660 | 매우 낮음 |
| 기본 절약 | ~6-8GB | RTX 3080, RTX 4070 | 낮음 |
| 8-bit 양자화 | ~4-6GB | RTX 3070, RTX 4060 | 중간 |
| 권장 설정 | ~8-10GB | RTX 3080Ti, RTX 4070Ti | 높음 |
| 고성능 | ~12GB+ | RTX 4080, RTX 4090 | 최고 |

### 4. 메모리 부족 시 단계적 해결책

#### 1단계: 기본 설정 조정
```bash
# 배치 크기 줄이기
--batch-size 1

# 시퀀스 길이 줄이기  
--max-length 256

# 양자화 사용
--use-4bit  # 또는 --use-8bit

# 샘플 수 제한
--limit-samples 500
```

#### 2단계: 극한 메모리 절약 모드
```bash
python train.py \
  --batch-size 1 \
  --max-length 128 \
  --lora-r 16 \
  --lora-alpha 32 \
  --use-8bit \
  --limit-samples 100
```

#### 3단계: 최소 메모리 모드 (train_fixed.py 사용)
```bash
python train_fixed.py \
  --batch_size 1 \
  --input_max_text_length 64 \
  --max_new_tokens 4 \
  --dataset_sample_size 5 \
  --max_ppo_steps 1 \
  --lora_r 4 \
  --lora_alpha 8 \
  --use_8bit_quantization \
  --enable_gradient_checkpointing \
  --max_memory_per_gpu 6GB
```

### 5. 메모리 모니터링 및 최적화 팁

#### GPU 메모리 사용량 실시간 확인
```bash
# 실시간 모니터링
watch -n 1 nvidia-smi

# 또는 Python에서 메모리 테스트
python debug_tools.py memory
```

#### 메모리 최적화 팁
1. **시퀀스 길이가 메모리에 가장 큰 영향**: `input_max_text_length` 우선 조정
2. **배치 크기 vs 성능**: 작은 배치는 메모리 절약하지만 훈련 안정성 저하 가능
3. **LoRA 설정**: `lora_r`이 클수록 더 많은 파라미터와 메모리 필요
4. **양자화 효과**: 8-bit는 메모리 50% 절약, 4-bit는 75% 절약하지만 약간의 성능 저하
5. **점진적 확장**: 작은 설정으로 시작해서 메모리 여유 확인 후 점진적 증가

## 🚨 포괄적 문제 해결 가이드

### 1. PPO 관련 오류 완전 해결

#### 'tuple' object has no attribute 'logits' 오류
**원인**: TRL이 내부적으로 생성하는 참조 모델이 양자화나 설정에 따라 tuple을 반환
**해결책**: `train_fixed.py` 사용 (FixedPPOTrainer 클래스 포함)

```bash
python train_fixed.py --use_8bit_quantization --enable_gradient_checkpointing
```

**기술적 해결 방법**:
- FixedPPOTrainer에서 참조 모델의 forward 메소드를 동적으로 패치
- tuple 출력을 ModelOutput으로 자동 변환
- 내부 참조 모델 자동 패치 시스템

#### PPOTrainer API 오류
**해결책**: 안정적인 SFT 대안 사용
```bash
python train.py --dataset-name maywell/korean_textbooks --use-8bit
```

### 2. 환경 문제 해결

#### CUDA 메모리 부족 (OOM)
```bash
# 1단계: 환경 진단 및 자동 복구
python fix_environment.py full-fix

# 2단계: 메모리 최적화 설정으로 재시도
python train.py --batch-size 1 --max-length 256 --use-4bit

# 3단계: 극한 절약 모드
python train_fixed.py --batch_size 1 --input_max_text_length 128 --use_8bit_quantization
```

#### CUDA/GPU 인식 실패
```bash
# GPU 환경 복구
python fix_environment.py install-pytorch

# 시스템 진단
python fix_environment.py diagnose

# CUDA 설치 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### 의존성 설치 오류
```bash
# Python 개발 헤더 설치
sudo apt install python3.12-dev

# bitsandbytes 재설치
pip uninstall bitsandbytes
pip install bitsandbytes

# 환경 재설정
./setup.sh
```

### 3. 모델 로딩 및 훈련 실패
```bash
# 안전한 추론 스크립트 사용
python inference_safe.py --interactive

# 모델 구조 분석
python analyze_model.py

# 환경 검증
python environment_validation.py
```

### 4. 성능 문제 해결

#### 훈련 속도가 너무 느린 경우
```bash
# 데이터셋 크기 줄이기
--dataset_sample_size 100
--limit-samples 500

# PPO 스텝 줄이기
--max_ppo_steps 2

# Gradient checkpointing 비활성화 (메모리 여유가 있다면)
# --enable_gradient_checkpointing 제거
```

#### 모델 품질이 낮은 경우
```bash
# 학습률 조정
--learning-rate 1e-4  # 또는 5e-5

# 에포크 증가
--epochs 5

# LoRA rank 증가 (메모리 허용 시)
--lora-r 128
--lora-alpha 256

# 더 큰 데이터셋 사용
--limit-samples 제거
```

## 📈 스크립트 선택 및 설정 최적화 가이드

### 1. GPU 메모리별 권장 워크플로우

#### 6GB 이하 GPU
```bash
# 1. 환경 확인
python environment_validation.py

# 2. 기본 훈련 (극한 절약)
python train.py --batch-size 1 --max-length 256 --use-8bit --limit-samples 500

# 3. 테스트
python inference_safe.py --interactive
```

#### 8GB GPU
```bash
# 1. 최적화된 훈련
python train.py --batch-size 2 --max-length 512 --use-4bit

# 또는 PPO 사용 시
python train_fixed.py --batch_size 2 --use_8bit_quantization
```

#### 12GB GPU (RTX 3060)
```bash
# 1. 고성능 최적화 훈련
python optimized_train.py --batch-size 8 --max-length 2048 --use-4bit

# 2. 추가 학습
python continue_train.py --mode continue --batch-size 4
```

### 2. 사용 목적별 권장 워크플로우

#### 처음 시도하는 사용자
```bash
# 1. 환경 검증 및 설정
python environment_validation.py
python fix_environment.py full-fix

# 2. 안전한 기본 훈련
python train.py --dataset-name maywell/korean_textbooks --use-8bit --limit-samples 1000

# 3. 결과 확인
python program_testing.py
python inference.py --interactive
```

#### 고급 사용자 (PPO 필요)
```bash
# 1. PPO 완전 해결 버전 사용
python train_fixed.py --use_8bit_quantization --enable_gradient_checkpointing

# 2. 고급 평가
python evaluate.py
```

#### 프로덕션 환경
```bash
# 1. 최적화된 고품질 훈련
python optimized_train.py --merge-and-save --epochs 5

# 2. 종합 평가
python evaluate.py

# 3. 추가 학습으로 지속 개선
python continue_train.py --mode continue --merge-and-save
```

### 3. 데이터셋 크기별 권장사항

| 데이터 크기 | 권장 스크립트 | 권장 설정 | 예상 시간 |
|-----------|-------------|----------|----------|
| < 1K 샘플 | train.py | epochs=3-5, limit-samples 제거 | 30분-1시간 |
| 1K-10K | train.py 또는 continue_train.py | epochs=3-5 | 1-3시간 |
| 10K-50K | optimized_train.py | epochs=3-5, gradient-accumulation 조정 | 3-8시간 |
| 50K+ | optimized_train.py | epochs=1-3, 효율적 배치 설정 | 8시간+ |

### 4. 하이퍼파라미터 튜닝 상세 가이드

#### 학습률 선택
| 목적 | 학습률 | 배치 크기 | 에포크 | LoRA r | 상황 |
|------|--------|----------|--------|--------|------|
| 빠른 테스트 | 2e-4 | 1-2 | 1-3 | 16-32 | 개념 검증용 |
| 안정적 학습 | 1e-4 | 2-4 | 3-5 | 64 | 일반적 사용 |
| 고품질 학습 | 5e-5 | 4-8 | 5-10 | 128 | 최고 품질 |
| 세밀 조정 | 1e-5 | 8-16 | 10+ | 256 | 프로덕션 레벨 |

#### LoRA 설정 최적화
- **lora_r**: 16 (기본) → 64 (표준) → 128 (고품질) → 256 (최고품질)
- **lora_alpha**: lora_r의 2배 권장 (16→32, 64→128, 128→256)
- **lora_dropout**: 0.05 (낮은 드롭아웃) → 0.1 (표준) → 0.2 (높은 정규화)

## 🎯 고급 사용 시나리오별 가이드

### 1. 빠른 테스트 및 프로토타이핑 (5-15분)
```bash
# 환경 검증
python environment_validation.py

# 빠른 기본 학습
python train.py --limit-samples 100 --epochs 1 --batch-size 2

# 즉석 테스트
python program_testing.py
python inference.py --prompt "이 영화는"
```

### 2. 프로덕션 레벨 고품질 파인튜닝 (2-6시간)
```bash
# 1. 종합 환경 준비
python environment_validation.py
python fix_environment.py full-fix

# 2. 데이터 수집 (필요 시)
python main.py --collect --web "https://your-domain-site.com" --web-crawl --max-pages 100

# 3. 고품질 최적화 파인튜닝
python optimized_train.py --data-path collected_data.jsonl \
                         --epochs 5 \
                         --batch-size 8 \
                         --max-length 2048 \
                         --lora-r 128 \
                         --lora-alpha 256 \
                         --use-4bit \
                         --merge-and-save

# 4. 종합 평가
python evaluate.py

# 5. 추가 최적화 (필요 시)
python continue_train.py --mode continue --data-path additional_data.jsonl
```

### 3. 커스텀 도메인 특화 모델 개발
```bash
# 1. 도메인 특화 데이터 수집
python main.py --collect \
               --web "domain-specific-site1.com" "domain-specific-site2.com" \
               --web-crawl --max-pages 50 \
               --pdf "domain-paper1.pdf" "domain-paper2.pdf" \
               --output "domain_data.jsonl"

# 2. 기본 모델 학습
python train.py --data-path domain_data.jsonl --epochs 3

# 3. 성능 평가 후 추가 학습
python continue_train.py --finetuned-model my_korean_finetuned_model \
                        --data-path additional_domain_data.jsonl \
                        --mode continue

# 4. 최종 모델 병합 및 저장
python continue_train.py --mode merge_and_new --merge-and-save
```

### 4. 문제 해결 및 디버깅 전용 워크플로우
```bash
# 1. 종합 진단
python debug_tools.py all

# 2. 환경 문제 해결
python fix_environment.py full-fix

# 3. 안전한 테스트 실행
python run_scripts.py cpu-train

# 4. 메모리 최적화 디버깅
python debug_tools.py memory

# 5. PPO 관련 문제 해결
python train_fixed.py --batch_size 1 --use_8bit_quantization
```

### 5. 지속적 모델 개선 시스템
```bash
# 1주차: 기본 모델 구축
python train.py --dataset-name maywell/korean_textbooks

# 2주차: 도메인 데이터 추가
python continue_train.py --mode continue --data-path week2_data.jsonl

# 3주차: 성능 최적화
python optimized_train.py --data-path consolidated_data.jsonl

# 매주: 성능 평가 및 모니터링
python evaluate.py
```

## 📄 시스템 요구사항 및 호환성

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
- BeautifulSoup4, newspaper3k, PyPDF2, pdfplumber (데이터 수집용)
- Supabase (선택사항)

### 테스트된 환경
- Ubuntu 24.04 + RTX 3060 12GB
- Python 3.12 + CUDA 12.1
- 모든 주요 기능 검증 완료

### 참고자료
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [PEFT](https://github.com/huggingface/peft)
- [EleutherAI Polyglot-Ko](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
- [NSMC Dataset](https://huggingface.co/datasets/nsmc)

---

## 🏁 마무리

이 프로젝트는 한국어 LLM 파인튜닝의 전체 파이프라인을 제공하는 종합적인 시스템입니다. 데이터 수집부터 환경 설정, 모델 훈련, 평가, 문제 해결까지 모든 과정이 통합되어 있어, 초보자부터 고급 사용자까지 누구나 쉽게 한국어 모델을 파인튜닝할 수 있습니다.

### 핵심 장점
✅ **완전 통합**: 28개 분산 파일 + 8개 분산 문서 → 3개 통합 도구 + 1개 통합 문서  
✅ **문제 해결**: PPO 'tuple' 오류 완전 해결 (FixedPPOTrainer)  
✅ **메모리 최적화**: RTX 3060에서도 13억 파라미터 모델 훈련 가능  
✅ **자동화**: 환경 진단부터 복구까지 완전 자동화  
✅ **확장성**: 웹/PDF/텍스트 등 다양한 데이터 소스 지원  
✅ **단계별 가이드**: 초보자부터 전문가까지 맞춤형 워크플로우  
✅ **완전한 문서화**: 모든 기능과 문제 해결 방법 상세 설명  

### 시작하기
1. **초보자**: `python environment_validation.py` → `python train.py --use-8bit`
2. **고급 사용자**: `python optimized_train.py` 또는 `python train_fixed.py`
3. **문제 해결**: `python debug_tools.py all` → `python fix_environment.py full-fix`

각 스크립트의 세부 옵션은 `python script_name.py --help`로 확인할 수 있습니다.

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.