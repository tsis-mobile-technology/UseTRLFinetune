"""
다양한 소스에서 수집한 데이터를 학습에 적합한 형태로 가공하는 모듈
"""
import os
import json
import logging
import random
from datetime import datetime
from datasets import Dataset, DatasetDict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_processor')

def preprocess_text(text, min_length=100, max_length=10000):
    """
    텍스트 전처리: 정리 및 길이 필터링
    
    Args:
        text (str): 전처리할 텍스트
        min_length (int): 최소 텍스트 길이
        max_length (int): 최대 텍스트 길이
        
    Returns:
        str: 전처리된 텍스트 또는 None (유효하지 않은 경우)
    """
    if not text or not isinstance(text, str):
        return None
    
    # 공백 정리
    text = text.strip()
    
    # 길이 확인
    if len(text) < min_length:
        return None
    
    # 최대 길이로 자르기
    if len(text) > max_length:
        text = text[:max_length]
    
    return text

def create_dataset_from_texts(texts, split_ratio=0.9):
    """
    텍스트 리스트에서 Hugging Face 데이터셋 생성
    
    Args:
        texts (list): 텍스트 리스트
        split_ratio (float): 학습/검증 분할 비율
        
    Returns:
        DatasetDict: 학습 및 검증 데이터셋
    """
    if not texts or len(texts) == 0:
        logger.error("유효한 텍스트가 없습니다.")
        return None
    
    # 데이터 형식 변환
    data = [{"text": text} for text in texts if text]
    
    if not data or len(data) == 0:
        logger.error("유효한 데이터가 없습니다.")
        return None
    
    # 셔플 및 분할
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    validation_data = data[split_idx:]
    
    # 데이터셋 생성
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    
    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })

def create_dataset_from_jsonl(jsonl_file, text_field="text", split_ratio=0.9):
    """
    JSONL 파일에서 Hugging Face 데이터셋 생성
    
    Args:
        jsonl_file (str): JSONL 파일 경로
        text_field (str): 텍스트가 있는 필드 이름
        split_ratio (float): 학습/검증 분할 비율
        
    Returns:
        DatasetDict: 학습 및 검증 데이터셋
    """
    if not os.path.exists(jsonl_file):
        logger.error(f"파일을 찾을 수 없습니다: {jsonl_file}")
        return None
    
    try:
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if text_field in item and item[text_field]:
                        # 텍스트 전처리
                        text = preprocess_text(item[text_field])
                        if text:
                            data.append({"text": text})
        
        if not data or len(data) == 0:
            logger.error("유효한 데이터가 없습니다.")
            return None
        
        # 셔플 및 분할
        random.shuffle(data)
        split_idx = int(len(data) * split_ratio)
        train_data = data[:split_idx]
        validation_data = data[split_idx:]
        
        # 데이터셋 생성
        train_dataset = Dataset.from_list(train_data)
        validation_dataset = Dataset.from_list(validation_data)
        
        logger.info(f"데이터셋 생성 완료: 학습 {len(train_data)}개, 검증 {len(validation_data)}개")
        
        return DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset
        })
        
    except Exception as e:
        logger.error(f"JSONL 파일 처리 중 오류 발생: {str(e)}")
        return None

def create_dataset_from_supabase_data(data, text_field="text", split_ratio=0.9):
    """
    Supabase에서 가져온 데이터로 Hugging Face 데이터셋 생성
    
    Args:
        data (list): Supabase에서 가져온 데이터 리스트
        text_field (str): 텍스트가 있는 필드 이름
        split_ratio (float): 학습/검증 분할 비율
        
    Returns:
        DatasetDict: 학습 및 검증 데이터셋
    """
    if not data or len(data) == 0:
        logger.error("유효한 데이터가 없습니다.")
        return None
    
    processed_data = []
    for item in data:
        if text_field in item and item[text_field]:
            # 텍스트 전처리
            text = preprocess_text(item[text_field])
            if text:
                processed_data.append({"text": text})
    
    if not processed_data or len(processed_data) == 0:
        logger.error("처리 후 유효한 데이터가 없습니다.")
        return None
    
    # 셔플 및 분할
    random.shuffle(processed_data)
    split_idx = int(len(processed_data) * split_ratio)
    train_data = processed_data[:split_idx]
    validation_data = processed_data[split_idx:]
    
    # 데이터셋 생성
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    
    logger.info(f"데이터셋 생성 완료: 학습 {len(train_data)}개, 검증 {len(validation_data)}개")
    
    return DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })

def append_texts_to_jsonl(texts, output_file):
    """
    텍스트 리스트를 기존 JSONL 파일에 추가
    
    Args:
        texts (list): 텍스트 리스트
        output_file (str): 출력 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    if not texts or len(texts) == 0:
        logger.error("유효한 텍스트가 없습니다.")
        return False
    
    try:
        # 기존 데이터 수 확인
        existing_count = 0
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        existing_count += 1
        
        # 데이터 추가 (어펜드 모드)
        with open(output_file, 'a', encoding='utf-8') as f:
            for text in texts:
                if text:
                    # JSON 직렬화하여 한 줄로 저장
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
        
        logger.info(f"{len(texts)}개 텍스트를 {output_file}에 추가했습니다. (기존: {existing_count}개, 새로운: {len(texts)}개, 총: {existing_count + len(texts)}개)")
        return True
        
    except Exception as e:
        logger.error(f"JSONL 파일 추가 중 오류 발생: {str(e)}")
        return False

def save_texts_to_jsonl(texts, output_file, append_mode=False):
    """
    텍스트 리스트를 JSONL 파일로 저장 (덮어쓰기 또는 추가 모드)
    
    Args:
        texts (list): 텍스트 리스트
        output_file (str): 출력 파일 경로
        append_mode (bool): True면 기존 파일에 추가, False면 덮어쓰기
        
    Returns:
        bool: 성공 여부
    """
    if append_mode:
        return append_texts_to_jsonl(texts, output_file)
    else:
        # 기존 코드 (save_texts_to_jsonl)
        if not texts or len(texts) == 0:
            logger.error("유효한 텍스트가 없습니다.")
            return False
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    if text:
                        # JSON 직렬화하여 한 줄로 저장
                        f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
            
            logger.info(f"{len(texts)}개 텍스트를 {output_file}에 저장했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"JSONL 파일 저장 중 오류 발생: {str(e)}")
            return False

def save_dataset_to_jsonl(dataset, output_file, text_field="text"):
    """
    Hugging Face 데이터셋을 JSONL 파일로 저장
    
    Args:
        dataset (Dataset): Hugging Face 데이터셋
        output_file (str): 출력 파일 경로
        text_field (str): 텍스트가 있는 필드 이름
        
    Returns:
        bool: 성공 여부
    """
    if not dataset or len(dataset) == 0:
        logger.error("유효한 데이터셋이 없습니다.")
        return False
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                if text_field in item and item[text_field]:
                    # JSON 직렬화하여 한 줄로 저장
                    f.write(json.dumps({text_field: item[text_field]}, ensure_ascii=False) + '\n')
        
        logger.info(f"{len(dataset)}개 항목을 {output_file}에 저장했습니다.")
        return True
        
    except Exception as e:
        logger.error(f"JSONL 파일 저장 중 오류 발생: {str(e)}")
        return False

def prepare_training_data(dataset, tokenizer, max_length=512, batch_size=8):
    """
    모델 훈련을 위한 데이터셋 준비
    
    Args:
        dataset (DatasetDict): Hugging Face 데이터셋
        tokenizer: Hugging Face 토크나이저
        max_length (int): 최대 시퀀스 길이
        batch_size (int): 배치 크기
        
    Returns:
        dict: 학습 및 검증용 데이터셋
    """
    if not dataset or 'train' not in dataset or 'validation' not in dataset:
        logger.error("유효한 데이터셋이 없습니다.")
        return None
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    
    # 데이터셋 토큰화
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"]
    )
    
    # 학습에 적합한 형식으로 설정
    tokenized_datasets.set_format("torch")
    
    logger.info("데이터셋 준비 완료")
    
    return {
        'train': tokenized_datasets['train'],
        'validation': tokenized_datasets['validation']
    }

if __name__ == "__main__":
    # 테스트 코드
    test_texts = [
        "이것은 한국어 언어 모델 학습을 위한 첫 번째 테스트 데이터입니다. 충분히 긴 텍스트가 필요합니다.",
        "두 번째 테스트 데이터입니다. 이 데이터는 모델이 다양한 문장 구조를 학습하는 데 도움이 됩니다.",
        "세 번째 테스트 데이터는 조금 더 길게 작성하여 모델이 문맥을 파악하는 능력을 향상시키기 위한 것입니다. 여러 문장으로 구성된 텍스트가 필요합니다."
    ]
    
    # 데이터셋 생성 테스트
    dataset = create_dataset_from_texts(test_texts)
    
    if dataset:
        print(f"학습 데이터셋 크기: {len(dataset['train'])}")
        print(f"검증 데이터셋 크기: {len(dataset['validation'])}")
        
        # 첫 번째 학습 데이터 출력
        print("첫 번째 학습 데이터:")
        print(dataset['train'][0]['text'])