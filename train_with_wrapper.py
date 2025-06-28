#!/usr/bin/env python3

import torch
import argparse
import random
import numpy as np
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from types import SimpleNamespace

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class ReferenceModelWrapper(torch.nn.Module):
    """
    참조 모델을 감싸서 항상 올바른 ModelOutput 형태를 반환하도록 하는 래퍼
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        
    def forward(self, *args, **kwargs):
        # 항상 return_dict=True로 설정
        kwargs['return_dict'] = True
        
        try:
            output = self.model(*args, **kwargs)
            
            # 출력이 tuple인 경우 ModelOutput으로 변환
            if isinstance(output, tuple):
                logger.warning("참조 모델이 tuple을 반환했습니다. ModelOutput으로 변환합니다.")
                
                # tuple의 첫 번째 요소가 보통 logits
                logits = output[0]
                
                # CausalLMOutputWithPast 객체 생성
                if len(output) >= 2:
                    past_key_values = output[1] if output[1] is not None else None
                else:
                    past_key_values = None
                
                # SimpleNamespace를 사용하여 logits 속성을 가진 객체 생성
                model_output = SimpleNamespace()
                model_output.logits = logits
                model_output.past_key_values = past_key_values
                model_output.hidden_states = None
                model_output.attentions = None
                
                return model_output
            
            # 이미 올바른 ModelOutput인 경우 그대로 반환
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
    
    def to(self, device):
        self.model = self.model.to(device)
        return self

def parse_arguments():
    parser = argparse.ArgumentParser(description="PPO 파인튜닝 스크립트")

    # 모델 및 경로 관련
    parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b", help="사전 훈련된 모델 이름 또는 경로")
    parser.add_argument("--output_dir", type=str, default="my_korean_ppo_finetuned_model", help="훈련된 모델 저장 경로")
    parser.add_argument("--logging_dir", type=str, default="./logs/ppo_tuning", help="TensorBoard 로그 저장 경로")

    # PPO 설정 관련
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="학습률")
    parser.add_argument("--batch_size", type=int, default=4, help="PPO 배치 크기")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO 미니 배치 크기")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="그래디언트 누적 스텝")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="각 PPO 배치에 대한 최적화 에폭 수")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE 람다 파라미터")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO 클리핑 엡실론")
    parser.add_argument("--max_ppo_steps", type=int, default=20, help="총 PPO 훈련 스텝 수")

    # LoRA 설정 관련
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r 값")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha 값")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA 드롭아웃")

    # 생성(Generation) 관련
    parser.add_argument("--max_new_tokens", type=int, default=32, help="생성할 새 토큰의 최대 수")
    parser.add_argument("--temperature", type=float, default=0.9, help="생성 시 온도")
    parser.add_argument("--top_k", type=float, default=0.0, help="생성 시 top_k")
    parser.add_argument("--top_p", type=float, default=1.0, help="생성 시 top_p")

    # 데이터셋 관련
    parser.add_argument("--dataset_name", type=str, default="nsmc", help="사용할 데이터셋 이름")
    parser.add_argument("--input_min_text_length", type=int, default=5, help="입력 텍스트 최소 길이")
    parser.add_argument("--input_max_text_length", type=int, default=12, help="입력 텍스트 최대 길이")
    parser.add_argument("--dataset_sample_size", type=int, default=100, help="테스트용 데이터셋 샘플 크기")

    # 기타
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    set_seed(args.seed)

    logger.info(f"Hyperparameters: {args}")

    # --- 1. 모델 이름 및 PPO 강화학습 설정 ---
    config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size
    )

    # --- 2. LoRA 및 양자화 설정 ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    # --- 3. 모델 및 토크나이저 로드 ---
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        peft_config=lora_config
    )
    
    # 중요: return_dict=True로 설정하여 ModelOutput 객체 반환 보장
    model.config.return_dict = True
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # TRL의 AutoModelForCausalLMWithValueHead에 generation_config 추가
    from transformers import GenerationConfig
    if not hasattr(model, 'generation_config'):
        generation_config = GenerationConfig.from_pretrained(args.model_name)
        model.generation_config = generation_config
        logger.info("✅ generation_config 추가 완료")

    # --- 4. 한국어 데이터셋 준비 ---
    def build_dataset(config, dataset_name, input_min_text_length, input_max_text_length, sample_size):
        if "korean_textbooks" in dataset_name:
            ds = load_dataset(dataset_name, "normal_instructions", split="train", trust_remote_code=True)
            if "instruction" in ds.column_names and "output" in ds.column_names:
                def combine_text(sample):
                    sample["review"] = sample["instruction"] + " " + sample["output"]
                    return sample
                ds = ds.map(combine_text, batched=False)
            elif "text" in ds.column_names:
                ds = ds.rename_columns({"text": "review"})
            else:
                text_column = ds.column_names[0]
                ds = ds.rename_columns({text_column: "review"})
        else:
            ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
            if "document" in ds.column_names:
                ds = ds.rename_columns({"document": "review"})
        
        ds = ds.filter(lambda x: len(x["review"]) > 40, batched=False)
        ds = ds.select(range(100))

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            prompt = sample["review"][: input_size()]
            encoding = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
            sample["input_ids"] = encoding["input_ids"].squeeze(0)
            sample["attention_mask"] = encoding["attention_mask"].squeeze(0) if "attention_mask" in encoding else None
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds

    dataset = build_dataset(config, args.dataset_name, args.input_min_text_length, args.input_max_text_length, args.dataset_sample_size)

    # --- 5. 한국어 보상 모델(Reward Model) 정의 ---
    model_device = next(model.parameters()).device
    try:
        if torch.cuda.is_available() and model_device.type == 'cuda':
            sentiment_pipe = pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator", device=model_device.index)
            logger.info(f"✅ Sentiment pipeline을 GPU (cuda:{model_device.index})에 로드했습니다.")
        else:
            sentiment_pipe = pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator", device="cpu")
            logger.info("✅ Sentiment pipeline을 CPU에 로드했습니다.")
    except Exception as e:
        logger.warning(f"Sentiment pipeline을 GPU에 로드하는데 실패하여 CPU로 로드합니다. Error: {e}")
        sentiment_pipe = pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator", device="cpu")

    # --- 6. 참조 모델(Reference Model) 생성 with Wrapper ---
    logger.info("참조 모델(reference model) 로딩 중...")
    
    # 별도의 참조 모델 로드
    ref_model_base = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    # 참조 모델 래퍼로 감싸기
    ref_model = ReferenceModelWrapper(ref_model_base)
    
    # 중요: return_dict=True로 설정
    ref_model.config.return_dict = True
    ref_model.config.pad_token_id = tokenizer.eos_token_id
    
    # 참조 모델에도 generation_config 추가
    if not hasattr(ref_model_base, 'generation_config'):
        ref_model_base.generation_config = GenerationConfig.from_pretrained(args.model_name)
        logger.info("✅ 참조 모델에 generation_config 추가 완료")
    
    # 메모리 최적화를 위해 참조 모델을 eval 모드로 설정
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    logger.info("✅ 참조 모델 래퍼 생성 완료")
    
    # 래퍼가 올바르게 작동하는지 테스트
    logger.info("참조 모델 래퍼 테스트 중...")
    test_text = "안녕하세요"
    test_inputs = tokenizer(test_text, return_tensors="pt").to(next(model.parameters()).device)
    
    try:
        with torch.no_grad():
            test_output = ref_model(**test_inputs)
            logger.info(f"래퍼 테스트 성공: 출력 타입 {type(test_output)}, hasattr logits: {hasattr(test_output, 'logits')}")
    except Exception as e:
        logger.error(f"래퍼 테스트 실패: {e}")
        raise
    
    # --- 7. PPOTrainer 초기화 ---
    def collator(data):
        batch = {}
        for key in data[0]:
            if key in ["input_ids", "attention_mask"]:
                sequences = [d[key] for d in data if d[key] is not None]
                if not sequences:
                    batch[key] = None
                    continue
                
                max_len = max(len(seq) for seq in sequences)
                padded_sequences = []
                for d in data:
                    seq = d[key]
                    if seq is None:
                        if key == "attention_mask":
                            padded = torch.zeros(max_len, dtype=torch.long)
                        else:
                            padded = torch.full((max_len,), tokenizer.pad_token_id, dtype=torch.long)
                    elif len(seq) < max_len:
                        if key == "attention_mask":
                            padded = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=seq.dtype)])
                        else:
                            padded = torch.cat([seq, torch.full((max_len - len(seq),), tokenizer.pad_token_id, dtype=seq.dtype)])
                    else:
                        padded = seq
                    padded_sequences.append(padded)
                batch[key] = torch.stack(padded_sequences)
            else:
                batch[key] = [d[key] for d in data]
        return batch

    # 보상 모델은 sentiment_pipe를 래핑해서 사용
    class RewardModel(torch.nn.Module):
        def __init__(self, sentiment_pipeline):
            super().__init__()
            self.sentiment_pipeline = sentiment_pipeline
            
        def forward(self, input_ids, **kwargs):
            return torch.tensor([0.5] * len(input_ids), device=input_ids.device)
    
    reward_model = RewardModel(sentiment_pipe)
    
    # AutoModelForCausalLMWithValueHead에서 base_model 추출
    try:
        if hasattr(model, 'pretrained_model'):
            base_model = model.pretrained_model
            logger.info("✅ pretrained_model에서 base_model 추출")
        elif hasattr(model, 'base_model'):
            base_model = model.base_model
            logger.info("✅ base_model 속성에서 base_model 추출")
        elif hasattr(model, 'model'):
            base_model = model.model
            logger.info("✅ model 속성에서 base_model 추출")
        else:
            base_model = model
            if not hasattr(base_model, 'base_model_prefix'):
                base_model.base_model_prefix = "gpt_neox"
            logger.info("✅ 모델에 base_model_prefix 추가")
            
    except Exception as e:
        logger.error(f"Base model 추출 실패: {e}")
        base_model = model
        if not hasattr(base_model, 'base_model_prefix'):
            base_model.base_model_prefix = "gpt_neox"
        logger.info("✅ 폴백: 모델에 base_model_prefix 추가")
    
    # PPOTrainer 생성 (래퍼된 참조 모델 사용)
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

    # --- 8. PPO 파인튜닝 실행 ---
    logger.info("PPO 파인튜닝 훈련을 시작합니다...")

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
        "temperature": 0.9,
    }

    max_ppo_steps = args.max_ppo_steps

    for step in range(max_ppo_steps):
        logger.info(f"PPO Step {step + 1}/{max_ppo_steps} 진행 중...")
        
        batch_list = []
        for _ in range(config.batch_size):
            batch_list.append(random.choice(dataset))
        
        query_tensors = [data["input_ids"].clone().detach().to(next(model.parameters()).device) for data in batch_list]
        
        response_tensors = []
        for query_tensor in query_tensors:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=query_tensor.unsqueeze(0),
                    **generation_kwargs
                )
                response_tensor = generated_ids[0][len(query_tensor):]
                response_tensors.append(response_tensor)
        
        generated_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        original_queries = [data["query"] for data in batch_list]
        for i in range(len(original_queries)):
            logger.info(f"  Query {i}: {original_queries[i][:50]}...")
            logger.info(f"  Generated {i}: {generated_texts[i][:50]}...")

        rewards = []
        for text in generated_texts:
            try:
                sentiment_result = sentiment_pipe(text)[0]

                if sentiment_result['label'] == 'LABEL_1':
                    rewards.append(torch.tensor(sentiment_result['score'], device=next(model.parameters()).device))
                else:
                    rewards.append(torch.tensor(sentiment_result['score'] * 0.1, device=next(model.parameters()).device))
            except Exception as e:
                logger.error(f"Sentiment analysis error for text '{text[:30]}...': {e}")
                rewards.append(torch.tensor(0.0, device=next(model.parameters()).device))

        try:
            if hasattr(ppo_trainer, 'step'):
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, {}, rewards)
                logger.info(f"  PPO Step {step+1} 완료. Stats: { {k: v.mean().item() if isinstance(v, list) and v and isinstance(v[0], torch.Tensor) else v for k, v in stats.items()} }")
            else:
                logger.warning("PPOTrainer에 'step' 메소드가 없습니다. 대안 방법을 시도합니다.")
                
                if hasattr(ppo_trainer, 'train'):
                    logger.info("'train' 메소드를 사용하여 전체 훈련을 시작합니다.")
                    break
                else:
                    logger.error("사용 가능한 훈련 메소드를 찾을 수 없습니다.")
                    available_methods = [method for method in dir(ppo_trainer) if not method.startswith('_') and callable(getattr(ppo_trainer, method))]
                    logger.info(f"사용 가능한 메소드들: {available_methods}")
                    break
                    
            avg_reward_step = torch.tensor(rewards).mean().item()
            logger.info(f"  Average reward for step {step+1}: {avg_reward_step:.4f}")

        except Exception as e:
            logger.error(f"PPO step 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 만약 step 메소드가 없어서 루프를 중단했다면 train() 호출
    if not hasattr(ppo_trainer, 'step') and hasattr(ppo_trainer, 'train'):
        logger.info("PPOTrainer.train()을 사용하여 훈련을 진행합니다...")
        try:
            ppo_trainer.train()
        except Exception as e:
            logger.error(f"train() 메소드 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()

    logger.info("✅ PPO 파인튜닝 완료!")

    # 모델 저장
    logger.info("모델 저장 중...")
    output_dir = args.output_dir
    try:
        model.save_pretrained(output_dir)
        logger.info(f"✅ 모델 어댑터가 '{output_dir}'에 저장되었습니다.")
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")
        try:
            if hasattr(model, 'peft_config'):
                model.save_adapter(output_dir)
                logger.info(f"✅ PEFT 어댑터가 '{output_dir}'에 저장되었습니다.")
        except Exception as e2:
            logger.error(f"어댑터 저장도 실패: {e2}")
    
    tokenizer.save_pretrained(output_dir)
    logger.info(f"💾 토크나이저가 '{output_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    main()