#!/usr/bin/env python3

import torch
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.models import create_reference_model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reference_model_output():
    """참조 모델의 출력 형태를 테스트하는 함수"""
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    # 양자화 설정
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # 원본 모델 로드
    logger.info("원본 모델 로딩 중...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model.config.return_dict = True
    logger.info(f"원본 모델 return_dict: {model.config.return_dict}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 방법 1: create_reference_model 사용
    logger.info("\n=== 방법 1: create_reference_model 사용 ===")
    ref_model_1 = create_reference_model(model)
    
    # 방법 1의 return_dict 설정 체크
    logger.info(f"create_reference_model return_dict: {ref_model_1.config.return_dict}")
    
    # 명시적으로 return_dict=True 설정
    ref_model_1.config.return_dict = True
    logger.info(f"명시적 설정 후 return_dict: {ref_model_1.config.return_dict}")
    
    # 방법 2: 별도 모델 로드
    logger.info("\n=== 방법 2: 별도 모델 로드 ===")
    ref_model_2 = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    ref_model_2.config.return_dict = True
    ref_model_2.eval()
    for param in ref_model_2.parameters():
        param.requires_grad = False
    
    logger.info(f"별도 로드 모델 return_dict: {ref_model_2.config.return_dict}")
    
    # 테스트 입력 생성
    test_text = "안녕하세요. 오늘 날씨가"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    attention_mask = inputs["attention_mask"].to(next(model.parameters()).device)
    
    logger.info(f"\n테스트 입력: {test_text}")
    logger.info(f"Input shape: {input_ids.shape}")
    
    # 각 모델의 출력 테스트
    models_to_test = [
        ("원본 모델", model),
        ("참조 모델 1 (create_reference_model)", ref_model_1),
        ("참조 모델 2 (별도 로드)", ref_model_2)
    ]
    
    for name, test_model in models_to_test:
        logger.info(f"\n=== {name} 테스트 ===")
        try:
            with torch.no_grad():
                # forward() 메소드 직접 호출
                if hasattr(test_model, 'pretrained_model'):
                    # AutoModelForCausalLMWithValueHead의 경우
                    output = test_model.pretrained_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                else:
                    # 일반 모델의 경우
                    output = test_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                
                logger.info(f"출력 타입: {type(output)}")
                logger.info(f"출력이 tuple인가: {isinstance(output, tuple)}")
                
                if hasattr(output, 'logits'):
                    logger.info(f"logits 속성 존재: True, shape: {output.logits.shape}")
                else:
                    logger.info("logits 속성 존재: False")
                    if isinstance(output, tuple):
                        logger.info(f"Tuple 길이: {len(output)}")
                        for i, item in enumerate(output):
                            logger.info(f"  Tuple[{i}] 타입: {type(item)}, shape: {getattr(item, 'shape', 'no shape')}")
                
                # generate() 메소드도 테스트
                logger.info(f"\n{name} generate() 테스트:")
                gen_output = test_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                logger.info(f"Generate 출력 타입: {type(gen_output)}")
                
        except Exception as e:
            logger.error(f"{name} 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 직접적인 forward 호출도 테스트
    logger.info("\n=== 직접적인 forward 호출 테스트 ===")
    try:
        with torch.no_grad():
            # 참조 모델 2에 대해 다양한 방법으로 호출
            
            # 방법 1: __call__ 메소드 (return_dict=True 명시)
            output1 = ref_model_2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logger.info(f"__call__ with return_dict=True: {type(output1)}, hasattr logits: {hasattr(output1, 'logits')}")
            
            # 방법 2: forward 메소드 직접 호출
            output2 = ref_model_2.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logger.info(f"forward with return_dict=True: {type(output2)}, hasattr logits: {hasattr(output2, 'logits')}")
            
            # 방법 3: config의 return_dict 무시하고 강제 설정
            original_return_dict = ref_model_2.config.return_dict
            ref_model_2.config.return_dict = True
            output3 = ref_model_2(input_ids=input_ids, attention_mask=attention_mask)
            logger.info(f"config.return_dict=True: {type(output3)}, hasattr logits: {hasattr(output3, 'logits')}")
            ref_model_2.config.return_dict = original_return_dict
            
    except Exception as e:
        logger.error(f"직접 호출 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reference_model_output()