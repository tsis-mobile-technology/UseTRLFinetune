# 한국어 언어 모델을 위한 맞춤형 데이터 수집 및 파인튜닝 시스템

이 프로젝트는 다양한 소스(웹페이지, PDF, 텍스트 파일)에서 텍스트 데이터를 수집하고, Supabase를 이용한 데이터 저장 및 관리 기능을 제공하며, 수집된 데이터로 한국어 언어 모델을 효과적으로 파인튜닝하는 시스템입니다.

## 주요 기능

1. **다양한 소스에서 텍스트 데이터 수집**
   - 웹페이지 URL에서 텍스트 추출
   - PDF 파일에서 텍스트 추출
   - 로컬 텍스트 파일 처리
   
2. **Supabase 통합**
   - 수집된 데이터를 Supabase 테이블에 저장
   - 메타데이터와 함께 구조화된 형태로 데이터 관리
   - 저장된 데이터를 학습용으로 내보내기
   
3. **효율적인 모델 파인튜닝**
   - 8비트 양자화와 LoRA를 통한 메모리 효율적 파인튜닝
   - 제한된 GPU 자원(RTX 3060)에서도 효과적인 학습
   - TRL 라이브러리를 활용한 보상 기반 학습

## 시작하기

### 1. 환경 설정

Python 가상 환경을 생성하고 필요한 패키지를 설치합니다.

```bash
# 가상 환경 생성 및 활성화
python3 -m venv korean-llm-env
source korean-llm-env/bin/activate

# 필요한 패키지 설치
pip install torch torchvision torchaudio
pip install transformers datasets accelerate trl peft bitsandbytes sentencepiece
pip install beautifulsoup4 newspaper3k PyPDF2 pdfplumber supabase
```

### 2. Supabase 설정 (선택 사항)

Supabase를 사용하려면 다음 환경 변수를 설정합니다.

```bash
export SUPABASE_URL="your-supabase-url"
export SUPABASE_KEY="your-supabase-api-key"
```

또는 명령줄 인자로 직접 제공할 수 있습니다.

### 3. 데이터 수집

다양한 소스에서 데이터를 수집할 수 있습니다.

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

# Supabase에 데이터 저장
python main.py --collect --web "https://ko.wikipedia.org/wiki/인공지능" --supabase --supabase-url "your-url" --supabase-key "your-key"

# Supabase에서 데이터 내보내기
python main.py --export-supabase "exported_data.jsonl" --supabase-url "your-url" --supabase-key "your-key"
```

#### 1DEPTH 웹 크롤링 기능

`--web-crawl` 옵션을 사용하면 지정된 URL에서 링크된 하위 페이지들을 자동으로 찾아서 텍스트를 추출합니다:

- `--max-pages`: 크롤링할 최대 페이지 수 (기본값: 20)
- `--crawl-delay`: 페이지 간 지연 시간 초 단위 (기본값: 1)

예시:
```bash
# news.hada.io 메인 페이지와 링크된 20개 하위 페이지에서 텍스트 추출
python main.py --collect --web "https://news.hada.io/show" --web-crawl --max-pages 20 --crawl-delay 2 --output "hada_news_data.jsonl"
```

### 4. 모델 파인튜닝

수집한 데이터로 한국어 언어 모델을 파인튜닝합니다.

```bash
# 기본 파인튜닝
python custom_train.py --data-path "training_data.jsonl" --output-dir "my_finetuned_model"

# 상세 옵션 지정
python custom_train.py --data-path "web_data.jsonl" --model-name "EleutherAI/polyglot-ko-1.3b" --output-dir "my_korean_finetuned_model" --batch-size 4 --epochs 3 --learning-rate 1.41e-5 --max-length 512 --use-reward-model
```

### 5. 데모 실행

다양한 기능을 한 번에 시연하는 데모 스크립트를 실행할 수 있습니다.

```bash
# 모든 기능 데모
python demo.py --demo-mode all --output-dir "demo_output"

# 웹 스크래핑만 데모
python demo.py --demo-mode web --output-dir "demo_output"

# 데이터 수집만 하고 훈련은 건너뛰기
python demo.py --demo-mode all --output-dir "demo_output" --skip-training
```

## 파일 구조

- `main.py`: 메인 스크립트 (데이터 수집 및 모델 훈련 조율)
- `custom_train.py`: 커스텀 데이터로 모델 파인튜닝
- `data_processor.py`: 데이터 가공 및 처리
- `demo.py`: 기능 데모 스크립트
- `data_collectors/`: 데이터 수집 모듈
  - `web_scraper.py`: 웹페이지 텍스트 추출
  - `pdf_extractor.py`: PDF 텍스트 추출
  - `supabase_manager.py`: Supabase 연동 및 관리

## 사용 예시

### 1. 웹페이지에서 데이터 수집하여 모델 훈련

```bash
# 단계 1: 웹페이지에서 데이터 수집
python main.py --collect --web "https://ko.wikipedia.org/wiki/인공지능" "https://ko.wikipedia.org/wiki/기계학습" "https://ko.wikipedia.org/wiki/자연어_처리" --output "web_training_data.jsonl"

# 단계 2: 수집된 데이터로 모델 훈련
python custom_train.py --data-path "web_training_data.jsonl" --output-dir "web_trained_model"
```

### 2. Supabase 활용

```bash
# 단계 1: 다양한 소스에서 데이터 수집하여 Supabase에 저장
python main.py --collect --web "https://ko.wikipedia.org/wiki/인공지능" --pdf "https://example.com/sample.pdf" --supabase

# 단계 2: Supabase에서 데이터 내보내기
python main.py --export-supabase "supabase_data.jsonl"

# 단계 3: 내보낸 데이터로 모델 훈련
python custom_train.py --data-path "supabase_data.jsonl" --output-dir "supabase_trained_model"
```

## 주의사항

1. 웹 스크래핑과 PDF 추출은 대상 사이트/파일의 구조에 따라 결과가 달라질 수 있습니다.
2. 저작권이 있는 콘텐츠의 사용에 주의하세요.
3. GPU 메모리 제한으로 인해 배치 크기와 모델 설정을 적절히 조정해야 할 수 있습니다.
4. 수집된 데이터의 품질이 모델 성능에 큰 영향을 미칩니다. 품질 관리에 신경써주세요.

## 추가 정보

- 8비트 양자화와 LoRA를 통해 12GB VRAM(RTX 3060)에서도 13억 파라미터 모델을 파인튜닝할 수 있습니다.
- 보상 모델을 활용하면 특정 스타일이나 품질의 텍스트 생성에 모델을 특화시킬 수 있습니다.
- PDF 추출은 PyPDF2와 pdfplumber를 모두 사용하여 최적의 결과를 얻습니다.
- 웹 스크래핑은 일반 웹페이지와 뉴스 기사를 구분하여 처리합니다.