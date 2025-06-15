import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. 양자화 설정을 포함하여 원본 베이스 모델 로드
base_model_name = "EleutherAI/polyglot-ko-1.3b"
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. 저장된 LoRA 어댑터(파인튜닝된 가중치)를 베이스 모델에 적용
# "my_korean_finetuned_model"는 ppo_trainer.save_pretrained()로 저장한 디렉토리입니다.
peft_model_id = "my_korean_finetuned_model"
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval() # 모델을 추론 모드로 설정

# 3. 테스트 프롬프트로 추론 실행
prompt = "이 영화는 정말" # 긍정적인 리뷰 시작 부분
inputs = tokenizer(prompt, return_tensors="pt")
# token_type_ids가 있다면 제거
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']
inputs = inputs.to("cuda")

print("--- 파인튜닝 모델 생성 결과 ---")
# 텍스트 생성
# 파인튜닝된 모델은 긍정적인 방향으로 문장을 완성할 확률이 높습니다.
with torch.no_grad():
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=64, # 생성할 최대 토큰 수
        repetition_penalty=2.0, # 반복 페널티
        do_sample=True, # 샘플링 사용
        top_k=50,
        top_p=0.95
    )

# 결과 디코딩 및 출력
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(result)

# 예시 출력:
# 이 영화는 정말 감동적이고 따뜻한 작품이었어요. 배우들의 섬세한 연기 덕분에 이야기에 깊이 몰입할 수 있었습니다.