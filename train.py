import torch
import argparse
import random
import numpy as np
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, GenerationConfig, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig
import traceback
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed_value):
    """모든 랜덤 시드를 고정하는 함수"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def clear_gpu_cache():
    """GPU 메모리 캐시를 정리하는 함수"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parse_arguments():
    """스크립트 실행을 위한 인자들을 파싱하는 함수"""
    parser = argparse.ArgumentParser(description="PPO 파인튜닝 스크립트")
    parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b", help="사전 훈련된 모델 이름 또는 경로")
    parser.add_argument("--output_dir", type=str, default="my_korean_ppo_finetuned_model", help="훈련된 모델 저장 경로")
    parser.add_argument("--logging_dir", type=str, default="./logs/ppo_tuning", help="TensorBoard 로그 저장 경로")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="학습률")
    parser.add_argument("--batch_size", type=int, default=2, help="PPO 배치 크기")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="PPO 미니 배치 크기")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="그래디언트 누적 스텝")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="각 PPO 배치에 대한 최적화 에폭 수")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE 람다 파라미터")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO 클리핑 엡실론")
    parser.add_argument("--max_ppo_steps", type=int, default=10, help="총 PPO 훈련 스텝 수")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r 값")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha 값")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA 드롭아웃")
    parser.add_argument("--max_new_tokens", type=int, default=8, help="생성할 새 토큰의 최대 수")
    parser.add_argument("--temperature", type=float, default=0.9, help="생성 시 온도")
    parser.add_argument("--top_k", type=float, default=0.0, help="생성 시 top_k")
    parser.add_argument("--top_p", type=float, default=1.0, help="생성 시 top_p")
    parser.add_argument("--dataset_name", type=str, default="maywell/korean_textbooks", help="사용할 데이터셋 이름")
    parser.add_argument("--input_min_text_length", type=int, default=5, help="입력 텍스트 최소 길이")
    parser.add_argument("--input_max_text_length", type=int, default=1024, help="입력 텍스트 최대 길이")
    parser.add_argument("--dataset_sample_size", type=int, default=20, help="테스트용 데이터셋 샘플 크기")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", help="메모리 절약을 위해 gradient checkpointing 활성화")
    parser.add_argument("--use_8bit_quantization", action="store_true", help="8bit 양자화 사용")
    parser.add_argument("--max_memory_per_gpu", type=str, default="8GB", help="GPU당 최대 메모리 사용량")
    args = parser.parse_args()
    return args

def build_dataset(tokenizer, dataset_name, input_min_text_length, input_max_text_length, sample_size):
    """데이터셋을 로드하고 PPO 훈련에 맞게 전처리하는 함수"""
    logger.info(f"'{dataset_name}' 데이터셋을 로드합니다...")
    if "korean_textbooks" in dataset_name:
        ds = load_dataset(dataset_name, "normal_instructions", split="train", trust_remote_code=True)
        if "instruction" in ds.column_names and "output" in ds.column_names:
            def combine_text(sample):
                instruction = sample.get("instruction", "") or ""
                output = sample.get("output", "") or ""
                sample["review"] = f"{instruction} {output}".strip()
                return sample
            ds = ds.map(combine_text)
        else:
            logger.warning("'instruction' 또는 'output' 컬럼을 찾을 수 없습니다. 'text' 또는 첫 번째 컬럼을 사용합니다.")
            if 'text' in ds.column_names:
                ds = ds.rename_column("text", "review")
            else:
                first_col = ds.column_names[0]
                ds = ds.rename_column(first_col, "review")
    else:
        ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
        if "document" in ds.column_names:
            ds = ds.rename_column("document", "review")
    
    ds = ds.filter(lambda x: x["review"] is not None and len(x["review"]) > 20)
    ds = ds.select(range(min(sample_size, len(ds))))

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["review"][: input_size()]
        encoding = tokenizer(prompt, truncation=True, max_length=input_max_text_length)
        sample["input_ids"] = torch.tensor(encoding["input_ids"])
        sample["attention_mask"] = torch.tensor(encoding["attention_mask"])
        sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        return sample

    ds = ds.map(tokenize)
    ds.set_format(type="torch")
    return ds

# PPOTrainer가 요구하는 형식에 맞춘 보상 모델 클래스
class RewardModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config, sentiment_pipeline, tokenizer):
        super().__init__(config)
        self.sentiment_pipeline = sentiment_pipeline
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, **kwargs):
        device = input_ids.device
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        pipe_outputs = self.sentiment_pipeline(texts, truncation=True, max_length=512)
        rewards = []
        for output in pipe_outputs:
            if output['label'] == 'LABEL_1': # 긍정
                rewards.append(torch.tensor(output['score'], device=device))
            else: # 부정
                rewards.append(torch.tensor(1.0 - output['score'], device=device))
        return None, torch.stack(rewards)


def main():
    args = parse_arguments()
    set_seed(args.seed)
    logger.info(f"Hyperparameters: {args}")
    
    # 호환성을 위해 빈 PPOConfig를 생성하고 속성을 수동으로 할당합니다.
    ppo_config = PPOConfig()
    ppo_config.learning_rate = args.learning_rate
    ppo_config.batch_size = args.batch_size
    ppo_config.mini_batch_size = args.mini_batch_size
    ppo_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    ppo_config.ppo_epochs = args.ppo_epochs
    ppo_config.lam = args.lam
    ppo_config.clip_epsilon = args.clip_epsilon
    ppo_config.log_with = "tensorboard"
    ppo_config.tracker_project_name = "ppo_korean_finetune"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.use_8bit_quantization else None

    # 정책 모델과 가치 모델을 함께 로드합니다.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory={0: args.max_memory_per_gpu},
        peft_config=lora_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # 참조 모델은 학습되지 않으므로, 양자화 없이 원본으로 로드합니다.
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        max_memory={0: args.max_memory_per_gpu},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    logger.info("✅ 참조(ref) 모델 생성 완료 (비양자화)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def collator(data):
        batch = {key: [d[key] for d in data] for key in data[0]}
        return tokenizer.pad(batch, padding=True, return_tensors="pt")

    dataset = build_dataset(tokenizer, args.dataset_name, args.input_min_text_length, args.input_max_text_length, args.dataset_sample_size)
    
    device = next(model.parameters()).device
    sentiment_pipe = pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator", device=device)
    logger.info(f"✅ Sentiment pipeline을 GPU ({device})에 로드했습니다.")
    
    # PPOTrainer 초기화
    # 최신 TRL은 reward_model을 직접 사용하지 않으므로, None으로 설정할 수 있습니다.
    # 대신 보상 계산은 학습 루프에서 수동으로 이루어집니다.
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    logger.info("✅ PPOTrainer 생성 성공!")
    logger.info("PPO 파인튜닝 훈련을 시작합니다...")
    
    # 생성 관련 설정
    generation_kwargs = {
        "min_length": -1,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
    }

    # 자동화된 학습 루프 실행
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= args.max_ppo_steps:
            break
        
        query_tensors = batch["input_ids"]
        
        # 모델로부터 응답 생성
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # 감성 분석을 통해 보상 계산
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, truncation=True, max_length=512)
        rewards = []
        for output in pipe_outputs:
            if output['label'] == 'LABEL_1':
                rewards.append(torch.tensor(output['score'], device=device))
            else:
                rewards.append(torch.tensor(-output['score'], device=device)) # 부정적인 경우 음수 보상

        # PPO 스텝 실행
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            logger.info(f"Step {epoch+1}/{args.max_ppo_steps} | Mean reward: {torch.mean(torch.tensor(rewards)).item():.4f}")
        except Exception as e:
            logger.error(f"PPO step 실행 중 오류 발생: {e}")
            traceback.print_exc()
            break

    logger.info("✅ PPO 파인튜닝 완료!")

    logger.info("모델 저장 중...")
    output_dir = args.output_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ 모델 어댑터와 토크나이저가 '{output_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    from tqdm import tqdm
    main()
