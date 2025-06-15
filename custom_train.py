"""
커스텀 데이터셋으로 한국어 언어 모델을 파인튜닝하는 스크립트
"""
import os
import argparse
import logging
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig
from torch.utils.data import DataLoader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('custom_train')

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='커스텀 데이터로 한국어 언어 모델 파인튜닝')
    
    parser.add_argument('--model-name', default='EleutherAI/polyglot-ko-1.3b', 
                        help='파인튜닝할 모델 이름')
    parser.add_argument('--data-path', required=True, 
                        help='훈련 데이터 JSONL 파일 경로')
    parser.add_argument('--output-dir', default='my_korean_finetuned_model', 
                        help='모델 저장 디렉토리')
    parser.add_argument('--batch-size', type=int, default=4, 
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='훈련 에포크 수')
    parser.add_argument('--learning-rate', type=float, default=1.41e-5, 
                        help='학습률')
    parser.add_argument('--max-length', type=int, default=512, 
                        help='최대 시퀀스 길이')
    parser.add_argument('--min-text-length', type=int, default=5, 
                        help='입력 텍스트의 최소 길이')
    parser.add_argument('--max-text-length', type=int, default=20, 
                        help='입력 텍스트의 최대 길이')
    parser.add_argument('--max-new-tokens', type=int, default=48, 
                        help='생성할 최대 토큰 수')
    parser.add_argument('--limit-samples', type=int, default=1000, 
                        help='훈련에 사용할 최대 샘플 수 (메모리 제한)')
    parser.add_argument('--use-reward-model', action='store_true', 
                        help='보상 모델 사용 여부')
    parser.add_argument('--reward-model-name', default='monologg/koelectra-base-v3-discriminator', 
                        help='한국어 보상 모델 이름')
    
    return parser.parse_args()

def build_dataset_from_jsonl(data_path, tokenizer, input_min_text_length=5, input_max_text_length=20, limit=None):
    """
    JSONL 파일에서 데이터셋 구축
    
    Args:
        data_path (str): JSONL 파일 경로
        tokenizer: 토크나이저
        input_min_text_length (int): 입력 텍스트의 최소 길이
        input_max_text_length (int): 입력 텍스트의 최대 길이
        limit (int, optional): 사용할 최대 샘플 수
        
    Returns:
        dataset: 구축된 데이터셋
    """
    logger.info(f"JSONL 파일에서 데이터셋 구축: {data_path}")
    
    # JSONL 파일 로드
    ds = load_dataset('json', data_files=data_path, split='train')
    
    # 필요한 경우 샘플 수 제한
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
        logger.info(f"데이터셋을 {limit}개 샘플로 제한")
    
    # 텍스트 필드 확인
    if 'text' not in ds.column_names:
        raise ValueError("데이터셋에 'text' 필드가 없습니다.")
    
    # 텍스트 길이 필터링
    ds = ds.filter(lambda x: len(x["text"]) > 40, batched=False)
    logger.info(f"필터링 후 데이터셋 크기: {len(ds)}")
    
    # 입력 크기 샘플러 생성
    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    
    def tokenize(sample):
        # 텍스트의 일부분을 입력으로 사용
        prompt = sample["text"][: input_size()]
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    
    # 토큰화 적용
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    
    logger.info(f"데이터셋 구축 완료: {len(ds)}개 샘플")
    return ds

def train_with_custom_data(args):
    """
    커스텀 데이터로 모델 파인튜닝
    
    Args:
        args: 명령줄 인자
    """
    logger.info("모델 파인튜닝 시작")
    
    # PPO 설정
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=max(2, args.batch_size // 2),
        gradient_accumulation_steps=2
    )
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    # 모델 및 토크나이저 로드
    logger.info(f"모델 로드: {args.model_name}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        peft_config=lora_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 데이터셋 구축
    dataset = build_dataset_from_jsonl(
        args.data_path, 
        tokenizer,
        args.min_text_length,
        args.max_text_length,
        args.limit_samples
    )
    
    # 데이터 로더 설정
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    dataloader = DataLoader(
        dataset,
        batch_size=ppo_config.batch_size,
        collate_fn=collator,
        shuffle=True
    )
    
    # 보상 모델 설정
    if args.use_reward_model:
        logger.info(f"보상 모델 로드: {args.reward_model_name}")
        sentiment_pipe = pipeline("sentiment-analysis", model=args.reward_model_name, device_map="auto")
        
        def reward_fn(texts):
            """긍정적인 감정을 보상하는 함수"""
            rewards = []
            for text in texts:
                try:
                    result = sentiment_pipe(text)
                    if result[0]['label'] == 'positive' or result[0]['label'] == 'LABEL_1':  # 긍정
                        rewards.append(torch.tensor(result[0]['score']))
                    else:  # 부정
                        rewards.append(torch.tensor(1.0 - result[0]['score']))
                except Exception as e:
                    logger.warning(f"보상 계산 오류: {e}")
                    rewards.append(torch.tensor(0.5))  # 오류 시 중간값
            return rewards
    else:
        # 간단한 길이 기반 보상 함수
        def reward_fn(texts):
            rewards = []
            for text in texts:
                # 텍스트 길이에 비례한 보상 (최대 1.0)
                reward = min(1.0, len(text) / 1000)
                rewards.append(torch.tensor(reward))
            return rewards
    
    # PPO 트레이너 설정
    # TRL 버전에 따라 PPOTrainer 생성 방법이 다름
    try:
        # 방법 1: 개별 인자로 전달
        ppo_trainer = PPOTrainer(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            dataset=dataset,
            ppo_config=ppo_config
        )
    except TypeError as e1:
        try:
            # 방법 2: 기본 인자만 사용
            ppo_trainer = PPOTrainer(
                model=model,
                ref_model=None,
                tokenizer=tokenizer
            )
        except TypeError as e2:
            try:
                # 방법 3: 최소한의 인자
                ppo_trainer = PPOTrainer(
                    model=model
                )
            except TypeError as e3:
                logger.error(f"PPOTrainer 초기화 실패: {e1}, {e2}, {e3}")
                raise e3
    
    # 생성 설정
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
    }
    
    # 학습 루프
    logger.info(f"{args.epochs}번의 에포크로 훈련 시작")
    
    for epoch in range(args.epochs):
        logger.info(f"에포크 {epoch+1}/{args.epochs} 시작")
        
        total_samples = 0
        total_reward = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # 메모리 제한으로 각 에포크에서 처리할 배치 수 제한
            if batch_idx >= 50:
                logger.info(f"  에포크 당 최대 배치 수 도달: 50")
                break
            
            logger.info(f"  배치 {batch_idx+1} 처리 중 ({len(batch['input_ids'])}개 샘플)")
            
            # 입력 텐서
            query_tensors = batch["input_ids"]
            
            # 텍스트 생성
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            # 생성된 텍스트 디코딩
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # 원본 쿼리 확인
            batch["query"] = batch["query"]
            
            # 결합된 텍스트 생성 (쿼리 + 응답)
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            
            # 샘플 로깅
            if batch_idx % 10 == 0:
                logger.info(f"    샘플: '{texts[0][:100]}...'")
            
            # 보상 계산
            rewards = reward_fn(texts)
            avg_reward = sum(r.item() for r in rewards) / len(rewards)
            total_reward += avg_reward
            total_samples += 1
            
            # PPO 스텝 실행
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            logger.info(f"  배치 {batch_idx+1} 완료: 평균 보상={avg_reward:.4f}")
        
        # 에포크 평균 보상
        if total_samples > 0:
            epoch_avg_reward = total_reward / total_samples
            logger.info(f"에포크 {epoch+1} 완료: 평균 보상={epoch_avg_reward:.4f}")
        else:
            logger.warning(f"에포크 {epoch+1}에서 처리된 샘플 없음")
    
    # 모델 저장
    logger.info(f"훈련된 모델 저장: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    ppo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 훈련 설정 저장
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    logger.info("모델 파인튜닝 완료")

def main():
    """메인 함수"""
    args = parse_args()
    
    # 데이터 파일 확인
    if not os.path.exists(args.data_path):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {args.data_path}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 훈련
    try:
        train_with_custom_data(args)
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()