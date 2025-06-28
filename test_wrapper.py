#!/usr/bin/env python3

import torch
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
from types import SimpleNamespace

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                
                # SimpleNamespace를 사용하여 logits 속성을 가진 객체 생성
                model_output = SimpleNamespace()
                model_output.logits = logits
                model_output.past_key_values = output[1] if len(output) >= 2 else None
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
    
    def eval(self):
        self.model.eval()
        return self
    
    def parameters(self):
        return self.model.parameters()

def test_wrapper():
    """래퍼를 테스트하는 함수"""
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    # 양자화 설정
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # 모델 로드
    logger.info("모델 로딩 중...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 래퍼로 감싸기
    wrapped_model = ReferenceModelWrapper(model)
    wrapped_model.config.return_dict = True
    wrapped_model.eval()
    
    # 테스트 입력
    test_text = "안녕하세요. 오늘 날씨가"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)
    attention_mask = inputs["attention_mask"].to(next(model.parameters()).device)
    
    logger.info(f"테스트 입력: {test_text}")
    logger.info(f"Input shape: {input_ids.shape}")
    
    # 원본 모델 테스트
    logger.info("\n=== 원본 모델 테스트 ===")
    try:
        with torch.no_grad():
            original_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logger.info(f"원본 모델 출력 타입: {type(original_output)}")
            logger.info(f"원본 모델 hasattr logits: {hasattr(original_output, 'logits')}")
            if hasattr(original_output, 'logits'):
                logger.info(f"원본 모델 logits shape: {original_output.logits.shape}")
    except Exception as e:
        logger.error(f"원본 모델 테스트 실패: {e}")
    
    # 래퍼 모델 테스트
    logger.info("\n=== 래퍼 모델 테스트 ===")
    try:
        with torch.no_grad():
            wrapped_output = wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logger.info(f"래퍼 모델 출력 타입: {type(wrapped_output)}")
            logger.info(f"래퍼 모델 hasattr logits: {hasattr(wrapped_output, 'logits')}")
            if hasattr(wrapped_output, 'logits'):
                logger.info(f"래퍼 모델 logits shape: {wrapped_output.logits.shape}")
                
                # logits 접근 테스트 (PPOTrainer에서 하는 것처럼)
                try:
                    context_length = input_ids.shape[1]
                    ref_logits = wrapped_output.logits[:, context_length - 1 : -1]
                    logger.info(f"✅ logits 슬라이싱 성공: {ref_logits.shape}")
                except Exception as slice_error:
                    logger.error(f"❌ logits 슬라이싱 실패: {slice_error}")
                    
    except Exception as e:
        logger.error(f"래퍼 모델 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 강제로 tuple 반환하도록 테스트
    logger.info("\n=== 강제 tuple 반환 테스트 ===")
    try:
        with torch.no_grad():
            # return_dict=False로 설정하여 tuple 반환 강제
            tuple_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False
            )
            logger.info(f"tuple 출력 타입: {type(tuple_output)}")
            logger.info(f"tuple 길이: {len(tuple_output)}")
            
            # 래퍼가 이를 올바르게 처리하는지 테스트
            class TupleReturningModel:
                def __init__(self, tuple_output):
                    self.tuple_output = tuple_output
                    self.config = model.config
                    
                def __call__(self, *args, **kwargs):
                    return self.tuple_output
                    
                def eval(self):
                    return self
                    
                def parameters(self):
                    return model.parameters()
            
            tuple_model = TupleReturningModel(tuple_output)
            wrapped_tuple_model = ReferenceModelWrapper(tuple_model)
            
            wrapped_tuple_output = wrapped_tuple_model(input_ids=input_ids, attention_mask=attention_mask)
            logger.info(f"래퍼된 tuple 출력 타입: {type(wrapped_tuple_output)}")
            logger.info(f"래퍼된 tuple hasattr logits: {hasattr(wrapped_tuple_output, 'logits')}")
            
            if hasattr(wrapped_tuple_output, 'logits'):
                logger.info(f"래퍼된 tuple logits shape: {wrapped_tuple_output.logits.shape}")
                # PPOTrainer 스타일 접근
                context_length = input_ids.shape[1]
                ref_logits = wrapped_tuple_output.logits[:, context_length - 1 : -1]
                logger.info(f"✅ 래퍼된 tuple logits 슬라이싱 성공: {ref_logits.shape}")
            
    except Exception as e:
        logger.error(f"강제 tuple 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n✅ 래퍼 테스트 완료!")

if __name__ == "__main__":
    test_wrapper()