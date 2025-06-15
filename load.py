from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. 양자화 설정: 최신 bitsandbytes 방식 적용
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# 2. 한국어 모델을 8비트 양자화로 로드 (VRAM 사용량 대폭 감소)
model_name = "EleutherAI/polyglot-ko-1.3b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"  # 자동으로 GPU에 모델을 할당
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # 패딩 토큰 설정

# 3. k-bit 학습을 위해 모델을 전처리
model = prepare_model_for_kbit_training(model)

# 4. LoRA 설정 정의
peft_config = LoraConfig(
    r=64, # LoRA의 차원 (rank)
    lora_alpha=16, # LoRA 스케일링 파라미터
    lora_dropout=0.1, # LoRA 레이어의 드롭아웃 비율
    bias="none",
    task_type="CAUSAL_LM" # 태스크 타입 지정
)

# 5. PEFT 모델 생성: 기존 모델에 LoRA 어댑터를 추가
model = get_peft_model(model, peft_config)

# 학습 가능한 파라미터 수 확인 (전체 파라미터의 1% 미만임을 확인)
model.print_trainable_parameters()
# 출력 예시: trainable params: 7,864,320 || all params: 1,321,502,720 || trainable%: 0.5951