"""
다양한 소스에서 수집한 데이터를 학습에 적합한 형태로 가공하는 모듈
"""
import os
import json
import logging
import random
from datetime import datetime
from datasets import Dataset, DatasetDict
import re
import unicodedata
import emoji

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
    
    # 향상된 텍스트 정리 및 정규화 적용
    cleaned_text = clean_and_normalize_text(text)

    if not cleaned_text: # 청소 후 텍스트가 비었을 수 있음
        return None

    # 길이 확인
    if len(cleaned_text) < min_length:
        return None
    
    # 최대 길이로 자르기
    if len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length]
    
    return cleaned_text

def clean_and_normalize_text(text: str, remove_emojis_flag: bool = True) -> str:
    """
    텍스트를 정규화하고 청소합니다.
    - 유니코드 정규화 (NFKC)
    - 제어 문자 제거 (일부 공백 문자는 유지)
    - 이모지 제거 또는 텍스트 변환 (현재는 제거)
    - 공백 정규화 (다중 공백, 탭, 줄바꿈 등)
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. 유니코드 정규화 (NFKC 선호: 호환성 및 더 강력한 정규화)
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception as e:
        logger.warning(f"유니코드 정규화 중 오류 발생: {e}. 원본 텍스트 사용.")

    # 2. 제어 문자 제거 (개행(\n), 탭(\t)은 유지하고 나머지는 제거 또는 공백으로 대체)
    # \x00-\x08, \x0b, \x0c, \x0e-\x1f, \x7f-\x9f (C0, C1 제어 문자 영역)
    # 개행과 탭을 제외한 제어문자를 공백으로 바꾼 후, 공백 정규화에서 처리
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # 3. 이모지 처리
    if remove_emojis_flag:
        text = emoji.replace_emoji(text, replace='') # 이모지 제거
        # text = emoji.demojize(text, delimiters=(":", ":")) # 텍스트 표현으로 변경 (예: ":smile:")

    # 4. 공백 정규화
    # 모든 종류의 공백 문자(스페이스, 탭, 개행, 캐리지 리턴, 폼 피드, 수직 탭 등)를 단일 스페이스로 변환
    text = re.sub(r'\s+', ' ', text)
    text = text.strip() # 양 끝 공백 최종 제거

    # (선택 사항) 특정 구두점 주변 공백 정규화 (예: 마침표, 쉼표 앞 공백 제거)
    # text = re.sub(r'\s([?.!,](?:\s|$))', r'\1', text)

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
    print("--- 기존 테스트 시작 ---")
    test_texts_original = [
        "이것은 한국어 언어 모델 학습을 위한 첫 번째 테스트 데이터입니다. 충분히 긴 텍스트가 필요합니다.",
        "두 번째 테스트 데이터입니다. 이 데이터는 모델이 다양한 문장 구조를 학습하는 데 도움이 됩니다.",
        "세 번째 테스트 데이터는 조금 더 길게 작성하여 모델이 문맥을 파악하는 능력을 향상시키기 위한 것입니다. 여러 문장으로 구성된 텍스트가 필요합니다.",
        "짧은 글입니다." # min_length 테스트용
    ]
    
    processed_for_dataset = [preprocess_text(text) for text in test_texts_original]
    processed_for_dataset = [text for text in processed_for_dataset if text] # None 제거

    dataset = create_dataset_from_texts(processed_for_dataset)
    
    if dataset:
        print(f"학습 데이터셋 크기: {len(dataset['train'])}")
        if len(dataset['validation']) > 0:
             print(f"검증 데이터셋 크기: {len(dataset['validation'])}")
             print("첫 번째 검증 데이터:", dataset['validation'][0]['text'] if len(dataset['validation'][0]['text']) > 0 else "비어 있음")
        else:
            print("검증 데이터셋이 비어 있습니다.")

        if len(dataset['train']) > 0:
            print("첫 번째 학습 데이터:", dataset['train'][0]['text'])
        else:
            print("학습 데이터셋이 비어 있습니다.")

    print("\n--- 새로운 clean_and_normalize_text 테스트 시작 ---")
    test_cases_normalization = {
        "기본": "이것은   \t 여러 공백과 \n줄바꿈이 있는 텍스트입니다.  ",
        "제어 문자": "텍스트\x00중간에\x08제어문자\x1f가 있습니다.",
        "이모지": "안녕하세요! 😊 좋은 하루 보내세요 👍🙏",
        "유니코드 (너비)": "ﾃｷｽﾄ가 있습니다.", # 반각 가타카나
        "유니코드 (합성)": "한글 자모 합치기: ㄱㅏㄴㅏㄷㅏ", # 자모 분리된 것
        "모두 포함": " 복잡한 텍스트 \t😅\n\x0c유니코드 ｶ타카나와 제어문자\x1e 포함!  ",
        "짧은 텍스트": "짧음", # 전처리 후 None이 될 수 있음
        "공백만": "   \n\t   ",
        "이모지만": "😁😂😃😄😅😆",
        "제어문자만": "\x01\x02\x03\x04\x05"
    }

    for name, text_input in test_cases_normalization.items():
        print(f"\n--- {name} ---")
        print(f"원본: '{text_input}' (길이: {len(text_input)})")

        cleaned_no_emoji = clean_and_normalize_text(text_input, remove_emojis_flag=True)
        print(f"정리됨 (이모지 제거): '{cleaned_no_emoji}' (길이: {len(cleaned_no_emoji)})")
        
        cleaned_with_emoji_text = clean_and_normalize_text(text_input, remove_emojis_flag=False) # emoji 라이브러리 기본은 제거이므로, False면 유지됨
                                                                                                # emoji.demojize를 쓰려면 clean_and_normalize_text 수정 필요
        print(f"정리됨 (이모지 유지): '{cleaned_with_emoji_text}' (길이: {len(cleaned_with_emoji_text)})")

        # preprocess_text를 통해 최소 길이 필터링까지 확인
        preprocessed_text = preprocess_text(text_input, min_length=5) # 최소 길이를 5로 설정하여 테스트
        print(f"최종 전처리 (min_length=5): '{preprocessed_text}'")

    print("\n--- preprocess_text 상세 테스트 (min_length=10) ---")
    test_preprocess_inputs = [
        "이것은 충분히 긴 한국어 텍스트입니다. 이모지 😂😂😂 와 함께합니다.", # 정상
        "너무 짧아요", # 짧아서 None
        "       \n\n\n         ", # 공백만 있어서 None (clean 후 empty)
        "😂🤣😅😄😁😆😊😋😎😍", # 이모지만 있어서 None (emoji 제거 후 empty)
        "12345\x01\x02\x03\x04\x0567890", # 제어문자 제거 후 길이 만족
        "1\x012\x023\x03" # 제어문자 제거 후 짧아서 None
    ]
    for i, text_input in enumerate(test_preprocess_inputs):
        output = preprocess_text(text_input, min_length=10)
        print(f"입력 {i+1}: '{text_input}'")
        print(f"  preprocess_text 결과: '{output}'\n")