"""
학습 데이터 수집 및 모델 훈련을 조율하는 메인 스크립트
"""
import os
import argparse
import logging
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer

# 자체 모듈 임포트
from data_collectors.web_scraper import extract_text_from_url, process_urls, crawl_website_depth_1
from data_collectors.pdf_extractor import extract_text_from_pdf_url, process_pdf_urls
from data_collectors.supabase_manager import SupabaseManager
from data_processor import (
    create_dataset_from_texts,
    create_dataset_from_jsonl,
    create_dataset_from_supabase_data,
    save_texts_to_jsonl,
    save_dataset_to_jsonl,
    prepare_training_data
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

def setup_argparse():
    """명령줄 인자 설정"""
    parser = argparse.ArgumentParser(description='학습 데이터 수집 및 모델 훈련')
    
    # 데이터 수집 관련 인자
    parser.add_argument('--collect', action='store_true', help='데이터 수집 모드 활성화')
    parser.add_argument('--web', nargs='+', help='텍스트를 추출할 웹페이지 URL 목록')
    parser.add_argument('--web-crawl', action='store_true', help='1DEPTH 웹 크롤링 모드 활성화')
    parser.add_argument('--max-pages', type=int, default=200, help='크롤링할 최대 페이지 수')
    parser.add_argument('--crawl-delay', type=int, default=1, help='크롤링 지연 시간 (초)')
    parser.add_argument('--pdf', nargs='+', help='텍스트를 추출할 PDF URL 목록')
    parser.add_argument('--text', nargs='+', help='학습에 사용할 텍스트 파일 경로 목록')
    
    # Supabase 관련 인자
    parser.add_argument('--supabase', action='store_true', help='Supabase 사용 활성화')
    parser.add_argument('--supabase-url', help='Supabase URL')
    parser.add_argument('--supabase-key', help='Supabase API 키')
    parser.add_argument('--export-supabase', help='Supabase 데이터를 JSONL로 내보낼 파일 경로')
    
    # 데이터 처리 관련 인자
    parser.add_argument('--output', default='training_data.jsonl', help='수집한 데이터를 저장할 JSONL 파일 경로')
    parser.add_argument('--append', action='store_true', help='기존 파일에 새로운 데이터를 추가 (덮어쓰기 하지 않음)')
    parser.add_argument('--min-length', type=int, default=100, help='최소 텍스트 길이')
    parser.add_argument('--max-length', type=int, default=10000, help='최대 텍스트 길이')
    
    # 모델 훈련 관련 인자
    parser.add_argument('--train', action='store_true', help='모델 훈련 모드 활성화')
    parser.add_argument('--model-name', default='EleutherAI/polyglot-ko-1.3b', help='사용할 모델 이름')
    parser.add_argument('--training-data', help='훈련에 사용할 JSONL 파일 경로')
    parser.add_argument('--batch-size', type=int, default=4, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=3, help='훈련 에포크 수')
    parser.add_argument('--learning-rate', type=float, default=1.41e-5, help='학습률')
    parser.add_argument('--output-dir', default='my_korean_finetuned_model', help='훈련된 모델을 저장할 디렉토리')
    
    return parser.parse_args()

def collect_data_from_web(urls, use_crawl=False, max_pages=20, delay=1):
    """웹페이지에서 데이터 수집"""
    if use_crawl:
        # 1DEPTH 크롤링 모드
        all_results = []
        all_texts = []
        
        for url in urls:
            logger.info(f"1DEPTH 크롤링 시작: {url}")
            crawl_results = crawl_website_depth_1(url, max_pages=max_pages, delay=delay)
            
            if crawl_results:
                all_results.extend(crawl_results)
                # 텍스트만 추출
                texts = [result['text'] for result in crawl_results if result and 'text' in result]
                all_texts.extend(texts)
                logger.info(f"{url}에서 {len(texts)}개 페이지 텍스트 추출")
        
        logger.info(f"총 {len(all_texts)}개 페이지에서 텍스트 추출 성공")
        return all_texts, all_results
    else:
        # 기본 모드 (단일 페이지)
        logger.info(f"{len(urls)}개 웹페이지에서 텍스트 추출 시작")
        results = process_urls(urls)
        
        # 추출된 텍스트만 리스트로 변환
        texts = [result['text'] for result in results if result and 'text' in result]
        logger.info(f"{len(texts)}개 웹페이지에서 텍스트 추출 성공")
        
        return texts, results

def collect_data_from_pdf(urls):
    """PDF에서 데이터 수집"""
    logger.info(f"{len(urls)}개 PDF에서 텍스트 추출 시작")
    results = process_pdf_urls(urls)
    
    # 추출된 텍스트만 리스트로 변환
    texts = [result['text'] for result in results if result and 'text' in result]
    logger.info(f"{len(texts)}개 PDF에서 텍스트 추출 성공")
    
    return texts, results

def collect_data_from_text_files(file_paths):
    """텍스트 파일에서 데이터 수집"""
    texts = []
    results = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)
                results.append({
                    'text': text,
                    'file_path': file_path,
                    'filename': os.path.basename(file_path),
                    'length': len(text)
                })
            logger.info(f"파일 '{file_path}'에서 {len(text)} 글자 추출")
        except Exception as e:
            logger.error(f"파일 '{file_path}' 읽기 오류: {str(e)}")
    
    logger.info(f"{len(texts)}개 텍스트 파일에서 데이터 추출 성공")
    return texts, results

def save_to_supabase(supabase_manager, web_results=None, pdf_results=None, text_results=None):
    """수집한 데이터를 Supabase에 저장"""
    # 테이블 생성
    supabase_manager.create_training_data_table()
    
    # 웹 데이터 저장
    web_count = 0
    if web_results:
        for result in web_results:
            if result and 'text' in result:
                if supabase_manager.insert_web_data(result):
                    web_count += 1
    
    # PDF 데이터 저장
    pdf_count = 0
    if pdf_results:
        for result in pdf_results:
            if result and 'text' in result:
                if supabase_manager.insert_pdf_data(result):
                    pdf_count += 1
    
    # 텍스트 파일 데이터 저장
    text_count = 0
    if text_results:
        for result in text_results:
            if result and 'text' in result:
                if supabase_manager.insert_text_data(
                    text=result['text'],
                    file_name=result.get('filename'),
                    metadata={'length': result.get('length'), 'source': 'text_file'}
                ):
                    text_count += 1
    
    logger.info(f"Supabase 저장 완료: 웹 {web_count}개, PDF {pdf_count}개, 텍스트 {text_count}개")

def train_model(args):
    """수집한 데이터로 모델 훈련"""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from peft import LoraConfig
    from torch.utils.data import DataLoader
    
    logger.info("모델 훈련 시작")
    
    # 훈련 데이터 로드
    logger.info(f"훈련 데이터 로드: {args.training_data}")
    dataset = create_dataset_from_jsonl(args.training_data)
    
    if not dataset:
        logger.error("데이터셋 생성 실패")
        return False
    
    logger.info(f"학습 데이터: {len(dataset['train'])}개, 검증 데이터: {len(dataset['validation'])}개")
    
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
    
    # 데이터셋 준비
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=8,
        remove_columns=["text"]
    )
    
    # 학습에 적합한 형식으로 설정
    tokenized_datasets.set_format("torch")
    
    # 데이터 로더 설정
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=ppo_config.batch_size,
        collate_fn=collator,
        shuffle=True
    )
    
    # 간단한 보상 모델 정의
    def simple_reward_fn(texts):
        """간단한 보상 함수: 텍스트 길이에 기반한 보상"""
        rewards = []
        for text in texts:
            # 보상 로직 (예: 길이에 비례한 보상)
            reward = min(1.0, len(text) / 1000)  # 최대 1.0
            rewards.append(torch.tensor(reward))
        return rewards
    
    # PPO 트레이너 설정
    ppo_trainer = PPOTrainer(
        config=ppo_config,  # 최신 라이브러리는 'ppo_config' 대신 'config' 사용
        model=model,
        ref_model=None,  # 8bit, PEFT 사용 시 ref_model은 None으로 설정
    )
    
    # 생성 설정
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
    }
    
    # 학습 루프
    logger.info(f"{args.epochs}번의 에포크로 훈련 시작")
    
    for epoch in range(args.epochs):
        logger.info(f"에포크 {epoch+1}/{args.epochs} 시작")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 이 예제에서는 각 에포크마다 10개 배치만 처리 (시간 절약)
            if batch_idx >= 10:
                break
            
            logger.info(f"  배치 {batch_idx+1}/10 처리 중")
            
            # 입력 텐서
            query_tensors = batch["input_ids"]
            
            # 텍스트 생성
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            # 생성된 텍스트 디코딩
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # 원본 쿼리 디코딩
            batch["query"] = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
            
            # 결합된 텍스트 생성 (쿼리 + 응답)
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            
            # 보상 계산
            rewards = simple_reward_fn(texts)
            
            # PPO 스텝 실행
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            logger.info(f"  배치 {batch_idx+1} 완료: 평균 보상={sum(rewards).item()/len(rewards):.4f}")
        
        logger.info(f"에포크 {epoch+1} 완료")
    
    # 모델 저장
    logger.info(f"훈련된 모델 저장: {args.output_dir}")
    ppo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("모델 훈련 완료")
    return True

def main():
    """메인 함수"""
    args = setup_argparse()
    
    # Supabase 매니저 초기화 (필요한 경우)
    supabase_manager = None
    if args.supabase or args.export_supabase:
        supabase_url = args.supabase_url or os.environ.get("SUPABASE_URL")
        supabase_key = args.supabase_key or os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase URL과 API 키가 필요합니다. --supabase-url, --supabase-key 옵션을 사용하거나 환경 변수를 설정하세요.")
            return
        
        supabase_manager = SupabaseManager(supabase_url, supabase_key)
    
    # 데이터 수집 모드
    if args.collect:
        all_texts = []
        web_results = None
        pdf_results = None
        text_results = None
        
        # 웹페이지에서 데이터 수집
        if args.web:
            web_texts, web_results = collect_data_from_web(
                args.web, 
                use_crawl=args.web_crawl, 
                max_pages=args.max_pages, 
                delay=args.crawl_delay
            )
            all_texts.extend(web_texts)
        
        # PDF에서 데이터 수집
        if args.pdf:
            pdf_texts, pdf_results = collect_data_from_pdf(args.pdf)
            all_texts.extend(pdf_texts)
        
        # 텍스트 파일에서 데이터 수집
        if args.text:
            text_texts, text_results = collect_data_from_text_files(args.text)
            all_texts.extend(text_texts)
        
        # Supabase에 데이터 저장
        if args.supabase and supabase_manager:
            save_to_supabase(supabase_manager, web_results, pdf_results, text_results)
        
        # 수집한 데이터를 파일로 저장
        if all_texts:
            output_path = args.output
            # append 모드 사용 (기본값을 True로 설정하여 항상 추가 모드)
            success = save_texts_to_jsonl(all_texts, output_path, append_mode=True)
            if success:
                if args.append or os.path.exists(output_path):
                    logger.info(f"수집한 데이터가 {output_path}에 추가되었습니다.")
                else:
                    logger.info(f"수집한 데이터가 {output_path}에 저장되었습니다.")
        else:
            logger.warning("수집된 데이터가 없습니다.")
    
    # Supabase 데이터 내보내기
    if args.export_supabase and supabase_manager:
        success = supabase_manager.export_data_to_jsonl(args.export_supabase)
        if success:
            logger.info(f"Supabase 데이터가 {args.export_supabase}에 내보내졌습니다.")
    
    # 모델 훈련 모드
    if args.train:
        # 훈련 데이터 파일 경로 확인
        training_data = args.training_data or args.output
        
        if not os.path.exists(training_data):
            logger.error(f"훈련 데이터 파일을 찾을 수 없습니다: {training_data}")
            return
        
        # 모델 훈련
        args.training_data = training_data
        train_success = train_model(args)
        
        if train_success:
            logger.info(f"모델이 성공적으로 훈련되어 {args.output_dir}에 저장되었습니다.")
        else:
            logger.error("모델 훈련 실패")

if __name__ == "__main__":
    main()