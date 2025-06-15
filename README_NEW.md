# 한국어 LLM TRL 파인튜닝 프로젝트

이 프로젝트는 **Ubuntu 24.04**와 **NVIDIA RTX 3060 (12GB VRAM)** 환경에서 Hugging Face의 TRL(Transformer Reinforcement Learning)을 사용하여 한국어 대규모 언어 모델을 효과적으로 파인튜닝하는 방법을 제공합니다.

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론 후 디렉터리 이동
cd UseTRLFinetune

# 설정 스크립트 실행 권한 부여
chmod +x setup.sh

# 자동 환경 설정
./setup.sh
```

### 2. 가상환경 활성화
```bash
source korean-llm-env/bin/activate
```

### 3. 모델 학습 (이미 완료된 상태)
```bash
python train.py
```

### 4. 모델 테스트
```bash
python test.py
```

## 📁 프로젝트 구조

```
UseTRLFinetune/
├── README.md                    # 프로젝트 문서
├── requirements.txt             # Python 의존성
├── setup.sh                     # 환경 설정 스크립트
├── train.py                     # 메인 학습 스크립트
├── test.py                      # 기본 테스트 스크립트
├── load.py                      # 모델 로딩 스크립트
├── evaluate.py                  # 고급 평가 스크립트
├── inference.py                 # 추론 및 대화형 스크립트
├── korean-llm-env/              # Python 가상환경
└── my_korean_finetuned_model/   # 파인튜닝된 모델 (LoRA 어댑터)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── pytorch_model.bin
    └── ...
```

## 🛠️ 스크립트 설명

### `train.py` - 메인 학습 스크립트
- **기능**: TRL PPO를 사용한 강화학습 기반 파인튜닝
- **데이터셋**: NSMC (네이버 영화 리뷰)
- **최적화**: LoRA + 8-bit 양자화로 RTX 3060 최적화
- **목표**: 긍정적인 영화 리뷰 생성 학습

### `test.py` - 기본 테스트
- **기능**: 파인튜닝된 모델로 간단한 텍스트 생성 테스트
- **용도**: 모델이 정상적으로 작동하는지 확인

### `evaluate.py` - 고급 평가 도구
- **기능**: 
  - 베이스 모델 vs 파인튜닝 모델 비교
  - 감정 분석 기반 성능 측정
  - 테스트 데이터셋 평가
  - 결과 JSON 저장

```bash
python evaluate.py
```

### `inference.py` - 추론 도구
- **기능**:
  - 단일 프롬프트 생성
  - 대화형 모드
  - 배치 테스트

```bash
# 대화형 모드
python inference.py --interactive

# 단일 프롬프트
python inference.py --prompt "이 영화는 정말" --max_tokens 100

# 기본 테스트 프롬프트들
python inference.py
```

## 🎯 모델 성능

### 파인튜닝 설정
- **베이스 모델**: EleutherAI/polyglot-ko-1.3b
- **방법**: TRL PPO (Proximal Policy Optimization)
- **최적화**: LoRA (rank=64) + 8-bit 양자화
- **보상 모델**: monologg/koelectra-base-v3-discriminator
- **타겟**: 긍정적인 한국어 텍스트 생성

### 메모리 최적화
- **양자화**: 8-bit (VRAM 사용량 1/4 감소)
- **PEFT**: LoRA로 학습 파라미터 0.6%만 사용
- **배치 크기**: RTX 3060에 최적화된 설정

## 📊 사용 예시

### 기본 텍스트 생성
```python
from inference import KoreanLLMInference

llm = KoreanLLMInference()
result = llm.generate("이 영화는 정말")
print(result)
# 출력: "이 영화는 정말 감동적이고 따뜻한 작품이었어요..."
```

### 모델 비교 평가
```python
from evaluate import KoreanLLMEvaluator

evaluator = KoreanLLMEvaluator()
results = evaluator.compare_models(["이 영화는", "배우들의 연기가"])
```

## 🔧 환경 요구사항

### 시스템 요구사항
- **OS**: Ubuntu 24.04 (다른 Linux 배포판도 가능)
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) 또는 유사한 성능
- **RAM**: 16GB 이상 권장
- **저장공간**: 10GB 이상

### Python 의존성
- Python 3.12+
- PyTorch 2.0+
- Transformers 4.30+
- TRL 0.7+
- PEFT 0.4+
- BitsAndBytes 0.41+

## 🚨 문제 해결

### 1. CUDA 관련 오류
```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 메모리 부족 오류
- `train.py`에서 `batch_size`를 더 작게 조정
- `generation_kwargs`에서 `max_new_tokens` 감소

### 3. bitsandbytes 설치 오류
```bash
# Python 개발 헤더 설치
sudo apt install python3.12-dev

# bitsandbytes 재설치
pip uninstall bitsandbytes
pip install bitsandbytes
```

## 📈 다음 단계

1. **더 큰 데이터셋**: 전체 NSMC 데이터셋 사용
2. **하이퍼파라미터 튜닝**: 학습률, LoRA rank 등 최적화
3. **다른 도메인**: 뉴스, 소설 등 다른 텍스트 도메인 적용
4. **모델 업그레이드**: 더 큰 한국어 모델 시도

## 📞 지원

문제가 발생하면 다음을 시도해보세요:

1. `setup.sh` 스크립트 재실행
2. 가상환경 재생성
3. GPU 메모리 확인 (`nvidia-smi`)
4. 로그 파일 확인

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 🙏 참고자료

- [Hugging Face TRL](https://github.com/huggingface/trl)
- [PEFT](https://github.com/huggingface/peft)
- [EleutherAI Polyglot-Ko](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
- [NSMC Dataset](https://huggingface.co/datasets/nsmc)