import torch
import argparse # argparse 추가
import random # random 추가
import numpy as np # numpy 추가
import logging # logging 추가
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig
# from torch.utils.data import DataLoader # PPOTrainer가 내부적으로 처리하므로 직접 사용 안 함

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

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
    parser.add_argument("--ppo_epochs", type=int, default=4, help="각 PPO 배치에 대한 최적화 에폭 수") # PPOConfig의 기본값은 4
    parser.add_argument("--lam", type=float, default=0.95, help="GAE 람다 파라미터") # PPOConfig의 기본값
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO 클리핑 엡실론") # PPOConfig의 기본값
    parser.add_argument("--max_ppo_steps", type=int, default=20, help="총 PPO 훈련 스텝 수") # 훈련 루프 반복 횟수

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
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        lam=args.lam,
        clip_epsilon=args.clip_epsilon,
        log_with="tensorboard",
        project_kwargs={"logging_dir": args.logging_dir}
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
        load_in_8bit=True # 현재는 8bit 양자화 고정
    )

    # --- 3. 모델 및 토크나이저 로드 ---
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        peft_config=lora_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 4. 한국어 데이터셋 준비 ---
    def build_dataset(config, dataset_name, input_min_text_length, input_max_text_length, sample_size):
        # """NSMC 데이터셋을 TRL 학습에 맞게 가공하는 함수"""
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
    # sentiment_pipe = pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator", device_map="auto")
    # device_map="auto"는 PPO 트레이너와 충돌 가능성 있음. device 명시 또는 CPU 사용 권장.
    # sentiment_pipe_device = model.device # 모델과 동일한 디바이스에 올리거나, CPU 사용 (-1)
    try:
        sentiment_pipe = pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator", device=model.device)
    except Exception:
        logger.warning("Sentiment pipeline을 모델과 동일한 디바이스에 로드하는데 실패하여 CPU로 로드합니다.")
        sentiment_pipe = pipeline("sentiment-analysis", model="monologg/koelectra-base-v3-discriminator")


    # --- 6. PPOTrainer 초기화 ---
    # PPOTrainer는 내부적으로 PPOConfig, 모델, 토크나이저, 데이터셋(옵션)을 사용합니다.
    # collator 함수는 PPOTrainer에 직접 전달할 수 있습니다.
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None, # 참조 모델 (옵션, 없으면 내부적으로 생성)
        tokenizer=tokenizer,
        dataset=dataset, # PPOTrainer가 내부적으로 DataLoader 생성
        data_collator=collator
    )

    # --- 7. PPO 파인튜닝 실행 ---
    logger.info("PPO 파인튜닝 훈련을 시작합니다...")

    # 생성 파라미터 설정 (PPOTrainer.generate에서 사용)
    generation_kwargs = {
        "min_length": -1, # 생성 시 최소 길이 (-1은 설정 안 함)
        "top_k": 0.0, # 상위 K 샘플링 (0.0은 사용 안 함)
        "top_p": 1.0, # 상위 P 샘플링 (1.0은 사용 안 함)
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32, # 생성할 새 토큰의 최대 수
        "temperature": 0.9,
    }

    # 훈련 루프 설정
    # num_epochs = 3 # 전체 데이터셋에 대한 반복 횟수 -> PPOTrainer는 스텝 기반으로 동작
    max_ppo_steps = 10 # PPO 스텝 수 (예시, 실제로는 더 많이 필요)

    for step in range(max_ppo_steps):
        logger.info(f"PPO Step {step + 1}/{max_ppo_steps} 진행 중...")
        
        # 데이터셋에서 배치 가져오기 (PPOTrainer가 내부적으로 처리)
        # 이 부분은 PPOTrainer의 내부 로직에 따라 dataset에서 샘플링 됩니다.
        # 사용자가 직접 dataloader를 만들 필요는 없지만, dataset은 준비되어야 합니다.
        # PPOTrainer는 config.batch_size에 맞춰 dataset에서 샘플링합니다.
        
        # build_dataset에서 ds.select(range(100))으로 데이터셋 크기를 줄였으므로,
        # config.batch_size에 따라 반복 횟수가 결정됩니다.
        # 예: 100개 샘플, batch_size=4 -> 25번의 내부 반복 후 1 PPO step 완료.
        # PPOTrainer는 내부적으로 for batch in self.dataloader: 루프를 돌며 query, response, reward를 수집합니다.
        # 우리는 이 루프를 직접 제어하는 대신, ppo_trainer.step()을 호출하여 한 PPO 스텝을 진행합니다.

        # 쿼리 텐서 가져오기 (dataset에서 자동으로 샘플링)
        # 실제로는 PPOTrainer가 내부적으로 dataset에서 batch_size만큼 샘플링하여 query_tensor를 만듭니다.
        # 여기서는 PPOTrainer의 step 함수에 직접 데이터를 전달하는 예시를 위해 수동으로 구성합니다.
        # 하지만 TRL의 일반적인 사용법은 PPOTrainer가 내부적으로 데이터를 처리하도록 하는 것입니다.
        
        # 현재 스크립트의 build_dataset은 'input_ids'와 'query'를 포함합니다.
        # PPOTrainer는 'input_ids'를 query_tensor로 사용하고, 'query'는 로깅/디버깅용입니다.
        
        batch_list = []
        for _ in range(config.batch_size):
            batch_list.append(random.choice(dataset)) # 데이터셋에서 무작위 샘플링
        
        query_tensors = [torch.tensor(data["input_ids"]).to(model.device) for data in batch_list]
        
        # 모델을 사용하여 응답 생성 (PPOTrainer의 generate 함수 사용)
        # response_tensors 리스트는 각 쿼리에 대한 생성된 응답 텐서들을 포함합니다.
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        
        # 생성된 텍스트 (디코딩된 문자열)
        # response_tensors는 생성된 부분만 포함 (prompt 제외)
        generated_texts = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # 원래 쿼리 텍스트 (로깅용)
        original_queries = [data["query"] for data in batch_list]
        for i in range(len(original_queries)):
            logger.info(f"  Query {i}: {original_queries[i][:50]}...")
            logger.info(f"  Generated {i}: {generated_texts[i][:50]}...")

        # 보상 계산
        rewards = []
        for text in generated_texts:
            try:
                # sentiment_pipe는 텍스트 리스트도 처리 가능
                # [{'label': 'LABEL_1', 'score': 0.99}] 형태의 리스트 반환
                # sentiment_results = sentiment_pipe(text)
                # sentiment_result = sentiment_results[0]
                sentiment_result = sentiment_pipe(text)[0] # 단일 텍스트 처리

                if sentiment_result['label'] == 'LABEL_1':  # 긍정
                    rewards.append(torch.tensor(sentiment_result['score'], device=model.device))
                else:  # 부정
                    rewards.append(torch.tensor(sentiment_result['score'] * 0.1, device=model.device)) # 부정 보상 대폭 감소
            except Exception as e:
                logger.error(f"Sentiment analysis error for text '{text[:30]}...': {e}")
                rewards.append(torch.tensor(0.0, device=model.device)) # 에러 시 0점 보상

        # PPO 스텝 실행 (모델 업데이트)
        # query_tensors: 입력 프롬프트 텐서 리스트
        # response_tensors: 모델이 생성한 응답 텐서 리스트
        # rewards: 각 응답에 대한 보상 텐서 리스트
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, {}, rewards) # 로그 기록
            logger.info(f"  PPO Step {step+1} 완료. Stats: { {k: v.mean().item() if isinstance(v, list) and v and isinstance(v[0], torch.Tensor) else v for k, v in stats.items()} }")
            avg_reward_step = torch.tensor(rewards).mean().item()
            logger.info(f"  Average reward for step {step+1}: {avg_reward_step:.4f}")

        except Exception as e:
            logger.error(f"PPO step 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 특정 상황 (예: 배치 크기 불일치)에서 오류 발생 가능

    logger.info("✅ PPO 파인튜닝 완료!")

    # 모델 저장
    logger.info("모델 저장 중...")
    # PPOTrainer는 내부적으로 모델을 저장하는 save_pretrained를 제공합니다.
    # 이 함수는 PEFT 어댑터만 저장하거나 전체 모델을 저장할 수 있습니다.
    # 일반적으로 LoRA 같은 PEFT를 사용하면 어댑터만 저장하는 것이 일반적입니다.
    # 전체 모델을 저장하려면 model.save_pretrained 사용 (내부적으로 PEFT 처리 가능성 있음)

    output_dir = "my_korean_ppo_finetuned_model"
    ppo_trainer.save_pretrained(output_dir) # 어댑터만 저장
    # model.save_pretrained(output_dir) # 전체 모델 저장 (PEFT 모델의 경우 내부적으로 어댑터만 저장될 수 있음)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"💾 모델과 토크나이저가 '{output_dir}'에 저장되었습니다.")

    # 만약 TensorBoard 로깅을 사용했다면, 다음 명령어로 확인 가능:
    # tensorboard --logdir logs/ppo_tuning