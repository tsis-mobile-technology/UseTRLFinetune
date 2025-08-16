# 메모리 최적화 PPO 훈련 가이드

## 🚀 해결된 문제
- **CUDA Out of Memory (OOM) 오류** 해결
- **'tuple' object has no attribute 'logits'** 오류 해결
- **메모리 효율적인 PPO 훈련** 구현

## 💡 주요 메모리 최적화 사항

### 1. 하이퍼파라미터 최적화
```bash
# 기본 메모리 절약 설정
--batch_size 2                    # 4 → 2 (50% 감소)
--mini_batch_size 1               # 2 → 1 (50% 감소)
--input_max_text_length 256       # 2048 → 256 (87% 감소)
--max_new_tokens 16              # 32 → 16 (50% 감소)
--dataset_sample_size 50         # 100 → 50 (50% 감소)
```

### 2. LoRA 파라미터 최적화
```bash
--lora_r 16                      # 128 → 16 (87% 감소)
--lora_alpha 32                  # 256 → 32 (87% 감소)
```

### 3. 메모리 최적화 기법
- **Gradient Checkpointing**: `--enable_gradient_checkpointing`
- **8-bit 양자화**: `--use_8bit_quantization`
- **GPU 메모리 제한**: `--max_memory_per_gpu 10GB`
- **Float16 정밀도**: 자동 적용 (양자화 미사용시)
- **메모리 캐시 정리**: 각 스텝마다 `torch.cuda.empty_cache()` 호출

### 4. 에러 처리 및 복구
- **OOM 에러 자동 복구**: 더 작은 설정으로 재시도
- **Attention Mask 처리**: 메모리 누수 방지
- **동적 배치 크기 조정**: 메모리 부족시 자동 감소

## 🔧 사용법

### 기본 메모리 절약 모드 (양자화 없음)
```bash
python train.py \
  --batch_size 1 \
  --mini_batch_size 1 \
  --input_max_text_length 128 \
  --max_new_tokens 8 \
  --dataset_sample_size 10 \
  --max_ppo_steps 2 \
  --lora_r 8 \
  --lora_alpha 16 \
  --enable_gradient_checkpointing \
  --max_memory_per_gpu 8GB
```

### 8-bit 양자화 모드 (더 많은 메모리 절약)
```bash
python train.py \
  --batch_size 2 \
  --mini_batch_size 1 \
  --input_max_text_length 256 \
  --max_new_tokens 16 \
  --dataset_sample_size 20 \
  --max_ppo_steps 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --use_8bit_quantization \
  --enable_gradient_checkpointing \
  --max_memory_per_gpu 10GB
```

### 최소 메모리 모드 (극한 절약)
```bash
python train.py \
  --batch_size 1 \
  --mini_batch_size 1 \
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

## 📊 메모리 사용량 추정

| 설정 | 예상 GPU 메모리 | 적합한 GPU |
|------|----------------|------------|
| 기본 절약 모드 | ~6-8GB | RTX 3080, RTX 4070 |
| 8-bit 양자화 모드 | ~4-6GB | RTX 3070, RTX 4060 |
| 최소 메모리 모드 | ~2-4GB | RTX 3060, GTX 1660 |

## 🛡️ 오류 해결

### 1. 여전히 OOM 오류가 발생하는 경우
- `--batch_size`를 1로 설정
- `--input_max_text_length`를 64 이하로 설정
- `--max_new_tokens`를 4 이하로 설정
- `--use_8bit_quantization` 플래그 추가

### 2. 'tuple' object has no attribute 'logits' 오류
- `--use_8bit_quantization` 플래그 제거
- `ref_model=None` 설정이 올바른지 확인 (코드에서 자동 처리)

### 3. 훈련 속도가 너무 느린 경우
- `--dataset_sample_size`를 더 작게 설정
- `--max_ppo_steps`를 줄여서 테스트
- Gradient checkpointing 비활성화 (메모리 여유가 있다면)

## 🔍 모니터링

### GPU 메모리 사용량 확인
```bash
# 실시간 모니터링
watch -n 1 nvidia-smi

# 또는 Python에서
python test_memory_optimized.py
```

### 훈련 로그 확인
```bash
# 메모리 관련 로그 필터링
python train.py [옵션들] 2>&1 | grep -i "memory\|cuda\|oom"
```

## 📝 추가 팁

1. **배치 크기 vs 성능**: 작은 배치 크기는 메모리를 절약하지만 훈련 안정성이 떨어질 수 있습니다.

2. **시퀀스 길이**: `input_max_text_length`가 메모리 사용량에 가장 큰 영향을 미칩니다.

3. **LoRA 설정**: `lora_r`이 클수록 더 많은 파라미터와 메모리가 필요합니다.

4. **양자화**: 8-bit 양자화는 메모리를 크게 절약하지만 약간의 성능 저하가 있을 수 있습니다.

5. **점진적 확장**: 작은 설정으로 시작해서 메모리 여유가 확인되면 점진적으로 늘려보세요.

## 🎯 권장 시작 설정

GPU 메모리가 11-12GB인 경우 (RTX 3080Ti, RTX 4080 등):
```bash
python train.py \
  --batch_size 2 \
  --input_max_text_length 256 \
  --max_new_tokens 16 \
  --lora_r 16 \
  --enable_gradient_checkpointing
```

이 설정으로 시작해서 메모리 사용량을 확인한 후 필요에 따라 조정하세요.