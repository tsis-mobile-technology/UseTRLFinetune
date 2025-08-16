#!/usr/bin/env python3
"""
프로젝트 주요 프로그램 테스트 통합 스크립트
- PPO 훈련 관련 테스트
- 참조 모델 수정 테스트
- 래퍼 솔루션 테스트
- 메모리 최적화 테스트
- 파인튜닝 효과 검증
"""

import os
import sys
import subprocess
import logging
import torch
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from peft import LoraConfig, PeftModel
from datasets import Dataset

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
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def eval(self):
        self.model.eval()
        return self
    
    def parameters(self):
        return self.model.parameters()

def test_create_reference_model():
    """create_reference_model 함수 테스트"""
    logger.info("=== create_reference_model 함수 테스트 ===")
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    try:
        # 토크나이저 생성
        logger.info("토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 설정
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # LoRA가 적용된 메인 모델 로드
        logger.info("LoRA 적용 메인 모델 로딩...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            peft_config=lora_config
        )
        
        if not hasattr(model, 'generation_config'):
            model.generation_config = GenerationConfig.from_pretrained(model_name)
        
        # create_reference_model로 참조 모델 생성
        logger.info("create_reference_model로 참조 모델 생성...")
        ref_model = create_reference_model(model)
        
        logger.info("✅ 두 모델 모두 성공적으로 로드됨")
        
        # 기본 모델 출력 테스트
        logger.info("모델 출력 테스트...")
        test_input = tokenizer("안녕하세요", return_tensors="pt", padding=True)
        
        # 같은 디바이스로 이동
        device = next(model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            # 메인 모델 테스트
            main_output = model(**test_input)
            logger.info(f"메인 모델 출력 타입: {type(main_output)}")
            if hasattr(main_output, 'logits'):
                logger.info(f"메인 모델 logits shape: {main_output.logits.shape}")
            
            # 참조 모델 테스트
            ref_output = ref_model(**test_input)
            logger.info(f"참조 모델 출력 타입: {type(ref_output)}")
            if hasattr(ref_output, 'logits'):
                logger.info(f"참조 모델 logits shape: {ref_output.logits.shape}")
            else:
                logger.error("❌ 참조 모델 출력에 logits 속성이 없습니다!")
                return False
        
        logger.info("✅ 두 모델 모두 logits를 가진 출력 생성")
        return True
        
    except Exception as e:
        logger.error(f"❌ create_reference_model 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_return_dict_fix():
    """return_dict 설정으로 tuple 문제 해결 테스트"""
    logger.info("=== return_dict 설정 테스트 ===")
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    try:
        # 토크나이저 생성
        logger.info("토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 설정
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # LoRA가 적용된 메인 모델 로드
        logger.info("LoRA 적용 메인 모델 로딩...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            peft_config=lora_config
        )
        
        # return_dict=True 설정
        model.config.return_dict = True
        
        if not hasattr(model, 'generation_config'):
            model.generation_config = GenerationConfig.from_pretrained(model_name)
        
        # 참조 모델 로드 (별도 로드, LoRA 없음)
        logger.info("참조 모델 로딩...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            # peft_config 없음
        )
        
        # 참조 모델에도 return_dict=True 설정
        ref_model.config.return_dict = True
        ref_model.config.pad_token_id = tokenizer.eos_token_id
        
        if not hasattr(ref_model, 'generation_config'):
            ref_model.generation_config = GenerationConfig.from_pretrained(model_name)
        
        # 참조 모델을 eval 모드로 설정
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
            
        logger.info("✅ 두 모델 모두 성공적으로 로드됨")
        
        # 모델 출력 테스트 - 핵심 테스트
        logger.info("모델 출력 테스트...")
        test_input = tokenizer("안녕하세요", return_tensors="pt", padding=True)
        
        # 같은 디바이스로 이동
        device = next(model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            # 메인 모델 테스트
            main_output = model(**test_input)
            logger.info(f"메인 모델 출력 타입: {type(main_output)}")
            logger.info(f"메인 모델 tuple 여부: {isinstance(main_output, tuple)}")
            if hasattr(main_output, 'logits'):
                logger.info(f"✅ 메인 모델 logits 속성 보유. Shape: {main_output.logits.shape}")
            else:
                logger.error("❌ 메인 모델 출력에 logits 속성이 없습니다!")
                return False
            
            # 참조 모델 테스트 - 여기서 오류가 발생했었음
            ref_output = ref_model(**test_input)
            logger.info(f"참조 모델 출력 타입: {type(ref_output)}")
            logger.info(f"참조 모델 tuple 여부: {isinstance(ref_output, tuple)}")
            if hasattr(ref_output, 'logits'):
                logger.info(f"✅ 참조 모델 logits 속성 보유. Shape: {ref_output.logits.shape}")
            else:
                logger.error("❌ 참조 모델 출력에 logits 속성이 없습니다!")
                return False
        
        logger.info("✅ 두 모델 모두 logits를 가진 ModelOutput 객체 생성")
        return True
        
    except Exception as e:
        logger.error(f"❌ return_dict 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wrapper_solution():
    """래퍼 솔루션 테스트"""
    logger.info("=== 래퍼 솔루션 테스트 ===")
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    try:
        # 양자화 설정
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 모델 로드
        logger.info("모델 로딩...")
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
        logger.info("원본 모델 테스트...")
        try:
            with torch.no_grad():
                original_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                logger.info(f"원본 모델 출력 타입: {type(original_output)}")
                logger.info(f"원본 모델 logits 속성 보유: {hasattr(original_output, 'logits')}")
                if hasattr(original_output, 'logits'):
                    logger.info(f"원본 모델 logits shape: {original_output.logits.shape}")
        except Exception as e:
            logger.error(f"원본 모델 테스트 실패: {e}")
        
        # 래퍼 모델 테스트
        logger.info("래퍼 모델 테스트...")
        try:
            with torch.no_grad():
                wrapped_output = wrapped_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logger.info(f"래퍼 모델 출력 타입: {type(wrapped_output)}")
                logger.info(f"래퍼 모델 logits 속성 보유: {hasattr(wrapped_output, 'logits')}")
                if hasattr(wrapped_output, 'logits'):
                    logger.info(f"래퍼 모델 logits shape: {wrapped_output.logits.shape}")
                    
                    # logits 접근 테스트 (PPOTrainer에서 하는 것처럼)
                    try:
                        context_length = input_ids.shape[1]
                        ref_logits = wrapped_output.logits[:, context_length - 1 : -1]
                        logger.info(f"✅ logits 슬라이싱 성공: {ref_logits.shape}")
                    except Exception as slice_error:
                        logger.error(f"❌ logits 슬라이싱 실패: {slice_error}")
                        return False
                        
        except Exception as e:
            logger.error(f"래퍼 모델 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info("✅ 래퍼 솔루션 테스트 통과!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 래퍼 솔루션 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_trainer_creation():
    """PPOTrainer 생성 테스트"""
    logger.info("=== PPOTrainer 생성 테스트 ===")
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    try:
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
        logger.info("메인 모델 로딩...")
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
        logger.info("참조 모델 생성...")
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
        return True
        
    except Exception as e:
        logger.error(f"❌ PPOTrainer 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization():
    """메모리 최적화 설정 테스트"""
    logger.info("=== 메모리 최적화 설정 테스트 ===")
    
    def check_gpu_memory():
        """GPU 메모리 상태 확인"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            cached_memory = torch.cuda.memory_reserved(0) / 1024**3
            free_memory = total_memory - cached_memory
            
            logger.info(f"GPU 메모리 상태:")
            logger.info(f"  총 메모리: {total_memory:.2f} GB")
            logger.info(f"  할당된 메모리: {allocated_memory:.2f} GB")
            logger.info(f"  캐시된 메모리: {cached_memory:.2f} GB")
            logger.info(f"  사용 가능 메모리: {free_memory:.2f} GB")
            
            return free_memory
        else:
            logger.warning("CUDA를 사용할 수 없습니다.")
            return 0
    
    try:
        # 초기 메모리 상태 확인
        initial_memory = check_gpu_memory()
        
        # 메모리 최적화된 설정들
        test_configs = [
            {
                "name": "기본 메모리 최적화",
                "settings": {
                    "batch_size": 1,
                    "mini_batch_size": 1,
                    "input_max_text_length": 128,
                    "max_new_tokens": 8,
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "use_8bit": True
                }
            },
            {
                "name": "더 작은 메모리 설정",
                "settings": {
                    "batch_size": 1,
                    "mini_batch_size": 1,
                    "input_max_text_length": 64,
                    "max_new_tokens": 4,
                    "lora_r": 4,
                    "lora_alpha": 8,
                    "use_8bit": True
                }
            }
        ]
        
        for config in test_configs:
            logger.info(f"\n--- {config['name']} 테스트 ---")
            settings = config['settings']
            
            # 메모리 상태 확인
            free_memory = check_gpu_memory()
            
            if free_memory < 2.0:  # 2GB 미만이면 경고
                logger.warning(f"사용 가능한 메모리가 부족합니다: {free_memory:.2f} GB")
                logger.info("더 작은 설정으로 조정이 필요합니다.")
            
            # 실제 훈련 명령어 생성 (실행하지 않고 출력만)
            train_cmd = [
                "python", "train.py",
                "--model_name", "EleutherAI/polyglot-ko-1.3b",
                "--batch_size", str(settings["batch_size"]),
                "--mini_batch_size", str(settings["mini_batch_size"]),
                "--input_max_text_length", str(settings["input_max_text_length"]),
                "--max_new_tokens", str(settings["max_new_tokens"]),
                "--lora_r", str(settings["lora_r"]),
                "--lora_alpha", str(settings["lora_alpha"]),
                "--dataset_sample_size", "10",
                "--max_ppo_steps", "2"
            ]
            
            if settings.get("use_8bit", False):
                train_cmd.append("--use_8bit_quantization")
            
            logger.info(f"권장 훈련 명령어:")
            logger.info(" ".join(train_cmd))
        
        logger.info("✅ 메모리 최적화 설정 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 메모리 최적화 테스트 실패: {e}")
        return False

def test_finetuning_effectiveness():
    """파인튜닝 효과 검증 테스트"""
    logger.info("=== 파인튜닝 효과 검증 테스트 ===")
    
    # 파인튜닝된 모델이 있는지 확인
    model_path = "my_korean_finetuned_model"
    if not os.path.exists(model_path):
        logger.warning(f"파인튜닝된 모델을 찾을 수 없습니다: {model_path}")
        logger.info("이 테스트는 파인튜닝 완료 후 실행해야 합니다.")
        return False
    
    try:
        # 양자화 설정
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 베이스 모델 로드
        logger.info("베이스 모델 로딩...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/polyglot-ko-1.3b",
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
        tokenizer.pad_token = tokenizer.eos_token
        
        # 파인튜닝된 모델 로드
        logger.info("파인튜닝된 모델 로딩...")
        finetuned_model = PeftModel.from_pretrained(base_model, model_path)
        finetuned_model.eval()
        
        logger.info("✅ 모델 로딩 완료!")
        
        # 테스트 프롬프트들
        test_prompts = [
            "이 영화는 정말",
            "배우들의 연기가",
            "스토리는",
            "감독의 연출이"
        ]
        
        generation_kwargs = {
            "max_new_tokens": 50,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 2.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        # 긍정적 키워드 리스트
        positive_words = ['좋', '훌륭', '멋지', '재미있', '감동', '완벽', '최고', '놀라운', '뛰어난', '추천']
        
        improvement_count = 0
        total_tests = len(test_prompts)
        
        for prompt in test_prompts:
            logger.info(f"\n프롬프트 테스트: '{prompt}'")
            
            # 입력 준비
            inputs = tokenizer(prompt, return_tensors="pt")
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            inputs = inputs.to("cuda")
            
            with torch.no_grad():
                # 베이스 모델 생성
                base_output = base_model.generate(**inputs, **generation_kwargs)
                base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
                
                # 파인튜닝된 모델 생성
                ft_output = finetuned_model.generate(**inputs, **generation_kwargs)
                ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)
            
            # 긍정 키워드 카운트
            base_positive_count = sum(1 for word in positive_words if word in base_text)
            ft_positive_count = sum(1 for word in positive_words if word in ft_text)
            
            logger.info(f"베이스 모델: {base_text}")
            logger.info(f"  긍정 키워드: {base_positive_count}개")
            
            logger.info(f"파인튜닝 모델: {ft_text}")
            logger.info(f"  긍정 키워드: {ft_positive_count}개")
            
            if ft_positive_count > base_positive_count:
                logger.info("  ✅ 파인튜닝 효과: 더 긍정적!")
                improvement_count += 1
            elif ft_positive_count == base_positive_count:
                logger.info("  ➖ 파인튜닝 효과: 비슷함")
            else:
                logger.info("  ❌ 파인튜닝 효과: 덜 긍정적")
        
        improvement_rate = improvement_count / total_tests
        logger.info(f"\n전체 파인튜닝 효과: {improvement_count}/{total_tests} ({improvement_rate*100:.1f}%) 개선")
        
        if improvement_rate >= 0.5:
            logger.info("✅ 파인튜닝이 효과적입니다!")
            return True
        else:
            logger.warning("⚠️ 파인튜닝 효과가 제한적입니다. 추가 튜닝이 필요할 수 있습니다.")
            return False
        
    except Exception as e:
        logger.error(f"❌ 파인튜닝 효과 검증 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_train_script():
    """train.py 스크립트 실행 테스트"""
    logger.info("=== train.py 스크립트 실행 테스트 ===")
    
    try:
        # 최소 매개변수로 테스트 실행
        cmd = [
            sys.executable, "train.py",
            "--model_name", "EleutherAI/polyglot-ko-1.3b",
            "--dataset_name", "maywell/korean_textbooks", 
            "--output_dir", "test_script_output",
            "--batch_size", "1",
            "--mini_batch_size", "1",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--input_max_text_length", "128",
            "--max_ppo_steps", "1",
            "--dataset_sample_size", "5",
            "--max_new_tokens", "4",
            "--use_8bit_quantization"
        ]
        
        logger.info("train.py 스크립트 실행 중...")
        logger.info("Command: " + " ".join(cmd))
        
        # 타임아웃을 설정하여 너무 오래 실행되지 않도록 함
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5분 타임아웃
        
        if result.returncode == 0:
            logger.info("✅ train.py 스크립트 실행 성공!")
            logger.info("STDOUT 마지막 부분:")
            logger.info(result.stdout[-500:])  # 마지막 500자만 출력
            return True
        else:
            logger.error("❌ train.py 스크립트 실행 실패!")
            logger.error("STDERR:")
            logger.error(result.stderr[-1000:])  # 마지막 1000자만 출력
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ train.py 스크립트 실행 시간 초과 (5분)")
        return False
    except Exception as e:
        logger.error(f"❌ train.py 스크립트 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    logger.info("🧪 프로젝트 주요 프로그램 테스트를 시작합니다...\n")
    
    tests = [
        ("create_reference_model 테스트", test_create_reference_model),
        ("return_dict 설정 테스트", test_return_dict_fix),
        ("래퍼 솔루션 테스트", test_wrapper_solution),
        ("PPOTrainer 생성 테스트", test_ppo_trainer_creation),
        ("메모리 최적화 테스트", test_memory_optimization),
        ("파인튜닝 효과 검증", test_finetuning_effectiveness),
        ("train.py 스크립트 테스트", test_train_script)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 {test_name} 시작")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} 성공!")
            else:
                logger.error(f"❌ {test_name} 실패!")
                
        except Exception as e:
            logger.error(f"❌ {test_name} 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    logger.info(f"\n{'='*60}")
    logger.info("🎯 프로그램 테스트 결과 요약")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{name:30s}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n전체 결과: {passed}/{total} 통과 ({passed/total*100:.1f}%)")
    
    # 권장사항
    if passed == total:
        logger.info("🎉 모든 주요 프로그램 테스트를 통과했습니다!")
        logger.info("프로젝트의 핵심 기능들이 정상적으로 작동합니다.")
    else:
        logger.warning("⚠️  일부 테스트가 실패했습니다.")
        logger.info("실패한 테스트들을 확인하여 문제를 해결해주세요.")
        
        failed_tests = [name for name, result in results if not result]
        logger.info("실패한 테스트들:")
        for test in failed_tests:
            logger.info(f"  - {test}")

if __name__ == "__main__":
    main()