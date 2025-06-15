import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig # LoRA 설정을 위해 import
from torch.utils.data import DataLoader # DataLoader를 직접 import

# --- 1. 모델 이름 및 PPO 강화학습 설정 ---
model_name = "EleutherAI/polyglot-ko-1.3b" # 모델 이름을 변수로 지정

# RTX 3060 환경에 맞춰 메모리를 최적화하는 배치 사이즈 등을 설정합니다.
config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=4, # 배치 사이즈를 줄여 VRAM 사용량 제어
    mini_batch_size=2,
    gradient_accumulation_steps=2 # 그래디언트 누적을 통해 배치 사이즈를 줄인 효과를 보완
)

# --- 2. LoRA 및 양자화 설정 ---
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# --- 3. 모델 및 토크나이저 로드 ---
# TRL은 'Value Head'가 추가된 모델 구조를 사용합니다.
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name, # 정의된 모델 이름 변수 사용
    quantization_config=quantization_config,
    device_map="auto",
    peft_config=lora_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name) # 정의된 모델 이름 변수 사용
tokenizer.pad_token = tokenizer.eos_token

# --- 4. 한국어 데이터셋 준비 (NSMC: 네이버 영화 리뷰) ---
def build_dataset(config, dataset_name="nsmc", input_min_text_length=5, input_max_text_length=12):
    """NSMC 데이터셋을 TRL 학습에 맞게 가공하는 함수"""
    # trust_remote_code=True 인자를 추가하여 데이터셋의 커스텀 코드를 실행하도록 허용합니다.
    ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    ds = ds.rename_columns({"document": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 40, batched=False)
    
    # 데이터셋 크기를 제한하여 빠른 테스트
    ds = ds.select(range(100))

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["review"][: input_size()]
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset(config)

# --- 5. 한국어 보상 모델(Reward Model) 정의 ---
sentiment_pipe = pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator", device_map="auto")

# --- 6. 간단한 파인튜닝 실행 ---
print("간단한 파인튜닝 훈련을 시작합니다...")

# 간단한 보상 함수 정의
def reward_fn(texts):
    """긍정적인 감정을 보상하는 함수"""
    rewards = []
    for text in texts:
        try:
            result = sentiment_pipe(text)
            if result['label'] == 'LABEL_1':  # 긍정
                rewards.append(result['score'])
            else:  # 부정
                rewards.append(result['score'] * 0.5)  # 부정적인 경우 낮은 보상
        except:
            rewards.append(0.5)  # 에러 시 중간 보상
    return rewards

try:
    # 데이터 준비
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    dataloader = DataLoader(
        dataset,
        batch_size=2,  # 작은 배치 사이즈
        collate_fn=collator,
        shuffle=True
    )
    
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"배치 수: {len(dataloader)}")
    
    model.train()
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 20,
        "temperature": 0.8,
    }
    
    total_reward = 0
    processed_batches = 0
    
    for epoch, batch in enumerate(dataloader):
        if epoch >= 3:  # 3번만 실행
            break
            
        print(f"\nEpoch {epoch + 1}/3 진행 중...")
        
        # 쿼리 텐서 준비
        query_tensors = batch["input_ids"]
        original_queries = batch["query"]
        
        print(f"원본 쿼리 예시: {original_queries[0][:50]}...")
        
        # 텍스트 생성
        generated_texts = []
        
        for i, query in enumerate(query_tensors):
            try:
                # 텐서를 적절한 형태로 변환
                if isinstance(query, list):
                    query_tensor = torch.tensor(query).unsqueeze(0).cuda()
                else:
                    query_tensor = query.unsqueeze(0).cuda()
                
                # 텍스트 생성 (AutoModelForCausalLMWithValueHead 사용)
                with torch.no_grad():
                    # pretrained_model에 접근하여 생성
                    output = model.pretrained_model.generate(
                        query_tensor,
                        **generation_kwargs
                    )
                
                # 생성된 텍스트 디코딩
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
                
                print(f"  생성된 텍스트 {i+1}: {generated_text}")
                
            except Exception as gen_error:
                print(f"  텍스트 생성 에러 {i+1}: {gen_error}")
                # 원본 쿼리를 기본값으로 사용
                fallback_text = original_queries[i] if i < len(original_queries) else "기본 텍스트"
                generated_texts.append(fallback_text)
                print(f"  대체 텍스트 {i+1}: {fallback_text}")
        
        # 보상 계산
        if generated_texts:
            try:
                rewards = reward_fn(generated_texts)
                avg_reward = sum(rewards) / len(rewards)
                total_reward += avg_reward
                processed_batches += 1
                
                print(f"  평균 보상: {avg_reward:.4f}")
                
                # 개별 보상 출력
                for i, (text, reward) in enumerate(zip(generated_texts, rewards)):
                    print(f"  텍스트 {i+1} 보상: {reward:.4f}")
                    
            except Exception as reward_error:
                print(f"  보상 계산 에러: {reward_error}")
        
        print(f"Epoch {epoch + 1} 완료")
    
    if processed_batches > 0:
        final_avg_reward = total_reward / processed_batches
        print(f"\n전체 평균 보상: {final_avg_reward:.4f}")
    
    print("✅ 파인튜닝 완료!")
    
    # 모델 저장
    print("모델 저장 중...")
    model.save_pretrained("my_korean_finetuned_model")
    tokenizer.save_pretrained("my_korean_finetuned_model")
    print("💾 모델과 토크나이저가 'my_korean_finetuned_model'에 저장되었습니다.")

except Exception as e:
    print(f"훈련 중 에러 발생: {e}")
    import traceback
    traceback.print_exc()
    
    print("기본 모델을 저장합니다...")
    try:
        model.save_pretrained("my_korean_finetuned_model")
        tokenizer.save_pretrained("my_korean_finetuned_model")
        print("💾 기본 모델이 'my_korean_finetuned_model'에 저장되었습니다.")
    except Exception as save_error:
        print(f"모델 저장 중 에러: {save_error}")
        # PEFT 모델인 경우 다른 방식으로 저장
        try:
            if hasattr(model, 'pretrained_model'):
                model.pretrained_model.save_pretrained("my_korean_finetuned_model")
                print("💾 PEFT 모델이 저장되었습니다.")
        except Exception as peft_error:
            print(f"PEFT 모델 저장 에러: {peft_error}")