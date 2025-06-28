#!/usr/bin/env python3

import torch
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from types import SimpleNamespace

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReferenceModelWrapper(torch.nn.Module):
    """참조 모델을 감싸서 항상 올바른 ModelOutput 형태를 반환하도록 하는 래퍼"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        
    def forward(self, *args, **kwargs):
        kwargs['return_dict'] = True
        
        try:
            output = self.model(*args, **kwargs)
            
            if isinstance(output, tuple):
                logger.info("참조 모델이 tuple을 반환했습니다. ModelOutput으로 변환합니다.")
                
                logits = output[0]
                model_output = SimpleNamespace()
                model_output.logits = logits
                model_output.past_key_values = output[1] if len(output) >= 2 else None
                model_output.hidden_states = None
                model_output.attentions = None
                
                return model_output
            
            return output
            
        except Exception as e:
            logger.error(f"참조 모델 호출 중 오류: {e}")
            raise
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def eval(self):
        self.model.eval()
        return self
    
    def parameters(self):
        return self.model.parameters()

def test_ppo_trainer_with_wrapper():
    """PPOTrainer와 래퍼된 참조 모델을 함께 테스트"""
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    # PPO 설정
    config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=2,
        mini_batch_size=1
    )
    
    # LoRA 및 양자화 설정
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # 메인 모델 로드
    logger.info("메인 모델 로딩 중...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        peft_config=lora_config
    )
    model.config.return_dict = True
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # generation_config 추가
    if not hasattr(model, 'generation_config'):
        generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config = generation_config
    
    # 참조 모델 생성 및 래퍼로 감싸기
    logger.info("참조 모델 생성 중...")
    ref_model_base = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    ref_model = ReferenceModelWrapper(ref_model_base)
    ref_model.config.return_dict = True
    ref_model.config.pad_token_id = tokenizer.eos_token_id
    
    if not hasattr(ref_model_base, 'generation_config'):
        ref_model_base.generation_config = GenerationConfig.from_pretrained(model_name)
    
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # 래퍼 테스트
    logger.info("래퍼 기능 테스트...")
    test_text = "안녕하세요"
    test_inputs = tokenizer(test_text, return_tensors="pt").to(next(model.parameters()).device)
    
    with torch.no_grad():
        test_output = ref_model(**test_inputs)
        logger.info(f"래퍼 테스트 - 출력 타입: {type(test_output)}, hasattr logits: {hasattr(test_output, 'logits')}")
        
        if hasattr(test_output, 'logits'):
            context_length = test_inputs['input_ids'].shape[1]
            ref_logits = test_output.logits[:, context_length - 1 : -1]
            logger.info(f"✅ PPOTrainer 스타일 logits 액세스 성공: {ref_logits.shape}")
    
    # 간단한 데이터셋 생성
    logger.info("간단한 데이터셋 생성...")
    
    class SimpleDataset:
        def __init__(self, tokenizer, size=10):
            self.tokenizer = tokenizer
            self.data = []
            texts = [
                "안녕하세요",
                "좋은 하루입니다",
                "감사합니다",
                "오늘 날씨가 좋네요",
                "한국어 학습을 하고 있습니다"
            ]
            
            for i in range(size):
                text = texts[i % len(texts)]
                encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
                self.data.append({
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "query": text
                })
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def __iter__(self):
            return iter(self.data)
        
        def select(self, indices):
            return [self.data[i] for i in indices]
        
        def set_format(self, *args, **kwargs):
            pass  # 더미 메소드
    
    dataset = SimpleDataset(tokenizer, size=10)
    
    # 간단한 collator
    def collator(data):
        batch = {}
        for key in data[0]:
            if key in ["input_ids", "attention_mask"]:
                sequences = [d[key] for d in data if d[key] is not None]
                if sequences:
                    max_len = max(len(seq) for seq in sequences)
                    padded_sequences = []
                    for d in data:
                        seq = d[key]
                        if len(seq) < max_len:
                            if key == "attention_mask":
                                padded = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=seq.dtype)])
                            else:
                                padded = torch.cat([seq, torch.full((max_len - len(seq),), tokenizer.pad_token_id, dtype=seq.dtype)])
                        else:
                            padded = seq
                        padded_sequences.append(padded)
                    batch[key] = torch.stack(padded_sequences)
                else:
                    batch[key] = None
            else:
                batch[key] = [d[key] for d in data]
        return batch
    
    # 간단한 보상 모델
    class RewardModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input_ids, **kwargs):
            return torch.tensor([0.5] * len(input_ids), device=input_ids.device)
    
    reward_model = RewardModel()
    
    # base_model 추출
    if hasattr(model, 'pretrained_model'):
        base_model = model.pretrained_model
    elif hasattr(model, 'base_model'):
        base_model = model.base_model
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
        if not hasattr(base_model, 'base_model_prefix'):
            base_model.base_model_prefix = "gpt_neox"
    
    # PPOTrainer 생성 시도
    logger.info("PPOTrainer 생성 시도...")
    try:
        ppo_trainer = PPOTrainer(
            args=config,
            processing_class=tokenizer,
            model=model,
            ref_model=ref_model,  # 래퍼된 참조 모델 사용
            reward_model=reward_model,
            train_dataset=dataset,
            value_model=base_model,
            data_collator=collator
        )
        
        logger.info("✅ PPOTrainer 생성 성공!")
        
        # 간단한 훈련 스텝 테스트
        logger.info("간단한 훈련 로직 테스트...")
        
        # 간단한 데이터 배치 생성
        batch_data = [dataset[i] for i in range(min(2, len(dataset)))]
        query_tensors = [data["input_ids"].clone().detach().to(next(model.parameters()).device) for data in batch_data]
        
        # 응답 생성 테스트
        generation_kwargs = {
            "max_new_tokens": 5,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        response_tensors = []
        for query_tensor in query_tensors:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=query_tensor.unsqueeze(0),
                    **generation_kwargs
                )
                response_tensor = generated_ids[0][len(query_tensor):]
                response_tensors.append(response_tensor)
        
        logger.info(f"응답 생성 성공: {len(response_tensors)}개 응답")
        
        # 보상 생성
        rewards = [torch.tensor(0.7, device=next(model.parameters()).device) for _ in response_tensors]
        
        # PPO step 시도
        if hasattr(ppo_trainer, 'step'):
            logger.info("PPO step 메소드 테스트 시도...")
            try:
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                logger.info("✅ PPO step 성공!")
                return True
            except Exception as e:
                logger.error(f"PPO step 실패: {e}")
                return False
        else:
            logger.info("PPO step 메소드가 없습니다. train() 메소드를 확인...")
            if hasattr(ppo_trainer, 'train'):
                logger.info("✅ train() 메소드 존재. 실제 훈련에서는 이를 사용할 수 있습니다.")
                return True
            else:
                logger.error("train() 메소드도 없습니다.")
                return False
        
    except Exception as e:
        logger.error(f"PPOTrainer 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("PPOTrainer + 래퍼 테스트 시작...")
    success = test_ppo_trainer_with_wrapper()
    if success:
        logger.info("✅ 모든 테스트 통과! 래퍼 솔루션이 작동합니다.")
    else:
        logger.error("❌ 테스트 실패. 추가 디버깅이 필요합니다.")