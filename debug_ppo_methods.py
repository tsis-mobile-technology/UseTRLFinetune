#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from transformers import BitsAndBytesConfig

# 간단한 PPOTrainer 생성하여 사용 가능한 메소드 확인
model_name = "EleutherAI/polyglot-ko-1.3b"

# 빠른 설정으로 작은 모델 로드
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print("모델 로드 중...")
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    peft_config=lora_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 더미 데이터셋
class DummyDataset:
    def __init__(self):
        self.data = [{"input_ids": torch.tensor([1, 2, 3])}]
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data[0]

dataset = DummyDataset()

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# 더미 reward model
class DummyRewardModel(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        return torch.tensor([0.5] * len(input_ids), device=input_ids.device)

reward_model = DummyRewardModel()

# base model 추출
if hasattr(model, 'pretrained_model'):
    base_model = model.pretrained_model
elif hasattr(model, 'base_model'):
    base_model = model.base_model
else:
    base_model = model

print("PPOTrainer 생성 중...")
try:
    ppo_trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=model,
        ref_model=None,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=base_model,
        data_collator=collator
    )
    print("✅ PPOTrainer 생성 성공!")
    
    print("\n사용 가능한 메소드들:")
    methods = [method for method in dir(ppo_trainer) if not method.startswith('_')]
    for method in sorted(methods):
        print(f"  - {method}")
        
    print(f"\n'step' 메소드 존재 여부: {'step' in methods}")
    print(f"'train' 메소드 존재 여부: {'train' in methods}")
    print(f"'generate' 메소드 존재 여부: {'generate' in methods}")
    
    # 실제 속성과 타입 확인
    if hasattr(ppo_trainer, 'step'):
        print(f"\nstep 메소드 타입: {type(ppo_trainer.step)}")
    else:
        print("\n❌ step 메소드가 존재하지 않습니다!")
        
    # 대안 메소드 확인
    if hasattr(ppo_trainer, 'train'):
        print(f"train 메소드 타입: {type(ppo_trainer.train)}")
    if hasattr(ppo_trainer, 'update'):
        print(f"update 메소드 타입: {type(ppo_trainer.update)}")
        
except Exception as e:
    print(f"❌ PPOTrainer 생성 실패: {e}")
    import traceback
    traceback.print_exc()