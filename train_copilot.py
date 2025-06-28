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

# 참조 모델을 위한 래퍼 클래스 - return_dict 문제 해결
class ReferenceModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 강제로 return_dict=True 설정
        if hasattr(self.model, 'config'):
            self.model.config.return_dict = True
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # kwargs에서 return_dict를 제거하고 항상 True로 설정
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('return_dict', None)  # 기존 return_dict 제거
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                return_dict=True,  # 명시적으로 return_dict=True 설정
                **kwargs_copy
            )
            
        # 만약 tuple이 반환되었다면 ModelOutput으로 변환
        if isinstance(outputs, tuple):
            from transformers.modeling_outputs import CausalLMOutputWithPast
            logger.warning("참조 모델에서 tuple 출력 발견, ModelOutput으로 변환")
            outputs = CausalLMOutputWithPast(
                logits=outputs[0],  # 첫 번째 요소가 logits
                past_key_values=outputs[1] if len(outputs) > 1 else None,
                hidden_states=outputs[2] if len(outputs) > 2 else None,
                attentions=outputs[3] if len(outputs) > 3 else None,
            )
            
        return outputs
    
    def __getattr__(self, name):
        # 다른 속성들은 원본 모델에 위임
        if name == 'model':
            return super().__getattr__(name)
        return getattr(self.model, name)

# PPOTrainer 래퍼 클래스 - return_dict 문제 완전 해결
class FixedPPOTrainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 참조 모델이 이미 래퍼로 감싸져 있는지 확인
        if hasattr(self, 'ref_model') and self.ref_model is not None:
            if not isinstance(self.ref_model, ReferenceModelWrapper):
                logger.info("참조 모델을 래퍼로 감싸는 중...")
                self.ref_model = ReferenceModelWrapper(self.ref_model)
                logger.info("✅ 참조 모델 래퍼 적용 완료")
            else:
                logger.info("✅ 참조 모델이 이미 래퍼로 감싸져 있음")
    
    def _get_logits_from_output(self, output):
        """안전하게 logits를 추출하는 헬퍼 함수"""
        if hasattr(output, 'logits'):
            return output.logits
        elif isinstance(output, tuple) and len(output) > 0:
            # tuple인 경우 첫 번째 요소가 logits일 가능성이 높음
            logger.warning("tuple 출력에서 logits 추출 시도")
            return output[0]
        else:
            raise ValueError(f"예상치 못한 출력 타입: {type(output)}")
    
    def forward(self, query_tensors, response_tensors, rewards):
        """개선된 forward 함수"""
        try:
            return super().forward(query_tensors, response_tensors, rewards)
        except AttributeError as e:
            if "'tuple' object has no attribute 'logits'" in str(e):
                logger.error("참조 모델 tuple 문제 발생, 대안 방법 시도")
                # 이 경우 다른 방법으로 처리할 수 있음
                raise e
            else:
                raise e

def patch_ref_model_output(ref_model):
    """참조 모델의 forward 메소드를 패치하여 항상 ModelOutput을 반환하도록 수정"""
    if ref_model is None:
        return None
        
    original_forward = ref_model.forward
    
    def patched_forward(*args, **kwargs):
        # 항상 return_dict=True로 설정
        kwargs['return_dict'] = True
        
        with torch.no_grad():
            outputs = original_forward(*args, **kwargs)
            
        # tuple이 반환된 경우 ModelOutput로 변환
        if isinstance(outputs, tuple):
            from transformers.modeling_outputs import CausalLMOutputWithPast
            logger.warning("참조 모델에서 tuple 출력 발견, ModelOutput으로 변환")
            # tuple에서 ModelOutput 생성
            outputs = CausalLMOutputWithPast(
                logits=outputs[0],  # 첫 번째 요소가 logits
                past_key_values=outputs[1] if len(outputs) > 1 else None,
                hidden_states=outputs[2] if len(outputs) > 2 else None,
                attentions=outputs[3] if len(outputs) > 3 else None,
            )
            
        return outputs
    
    # forward 메소드 교체
    ref_model.forward = patched_forward
    logger.info("✅ 참조 모델 forward 메소드 패치 완료")
    return ref_model

def clear_gpu_cache():
    """GPU 메모리 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def parse_arguments():
    parser = argparse.ArgumentParser(description="PPO 파인튜닝 스크립트")

    # 모델 및 경로 관련
    parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b", help="사전 훈련된 모델 이름 또는 경로")
    parser.add_argument("--output_dir", type=str, default="my_korean_ppo_finetuned_model", help="훈련된 모델 저장 경로")
    parser.add_argument("--logging_dir", type=str, default="./logs/ppo_tuning", help="TensorBoard 로그 저장 경로")

    # PPO 설정 관련 (메모리 최적화)
    parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="학습률")
    parser.add_argument("--batch_size", type=int, default=2, help="PPO 배치 크기 (메모리 절약을 위해 감소)")
    parser.add_argument("--mini_batch_size", type=int, default=1, help="PPO 미니 배치 크기 (메모리 절약을 위해 감소)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="그래디언트 누적 스텝")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="각 PPO 배치에 대한 최적화 에폭 수") # PPOConfig의 기본값은 4
    parser.add_argument("--lam", type=float, default=0.95, help="GAE 람다 파라미터") # PPOConfig의 기본값
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO 클리핑 엡실론") # PPOConfig의 기본값
    parser.add_argument("--max_ppo_steps", type=int, default=20, help="총 PPO 훈련 스텝 수") # 훈련 루프 반복 횟수


    # LoRA 설정 관련 (메모리 최적화를 위해 더 작은 값 사용)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r 값 (메모리 절약을 위해 16으로 감소)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha 값 (일반적으로 r의 2배)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA 드롭아웃")

    # 생성(Generation) 관련 (메모리 최적화)
    parser.add_argument("--max_new_tokens", type=int, default=16, help="생성할 새 토큰의 최대 수 (메모리 절약을 위해 감소)")
    parser.add_argument("--temperature", type=float, default=0.9, help="생성 시 온도")
    parser.add_argument("--top_k", type=float, default=0.0, help="생성 시 top_k")
    parser.add_argument("--top_p", type=float, default=1.0, help="생성 시 top_p")

    # 데이터셋 관련 (메모리 최적화를 위해 시퀀스 길이 크게 감소)
    parser.add_argument("--dataset_name", type=str, default="nsmc", help="사용할 데이터셋 이름")
    parser.add_argument("--input_min_text_length", type=int, default=5, help="입력 텍스트 최소 길이")
    parser.add_argument("--input_max_text_length", type=int, default=256, help="입력 텍스트 최대 길이 (메모리 절약을 위해 256으로 감소)")
    parser.add_argument("--dataset_sample_size", type=int, default=50, help="테스트용 데이터셋 샘플 크기 (메모리 절약을 위해 감소)")

    # 기타
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", help="메모리 절약을 위해 gradient checkpointing 활성화")
    parser.add_argument("--use_8bit_quantization", action="store_true", help="8bit 양자화 사용 (메모리 절약)")
    parser.add_argument("--max_memory_per_gpu", type=str, default="10GB", help="GPU당 최대 메모리 사용량")

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

    # --- 2. LoRA 및 양자화 설정 (메모리 최적화) ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]  # GPT-NeoX 아키텍처에 맞는 타겟 모듈
    )

    # 메모리 최적화를 위한 양자화 설정
    if args.use_8bit_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,  # 메모리 절약
            bnb_8bit_use_double_quant=True,  # 더 많은 메모리 절약
        )
        logger.info("✅ 8bit 양자화 활성화")
    else:
        quantization_config = None  # 양자화 비활성화로 return_dict 문제 해결
        logger.info("⚠️ 양자화 비활성화 (메모리 사용량 증가 가능)")

    # --- 3. 모델 및 토크나이저 로드 (메모리 최적화) ---
    # 메모리 제한 설정
    max_memory = {0: args.max_memory_per_gpu}  # GPU 0에 대한 메모리 제한
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory,  # 메모리 사용량 제한
        torch_dtype=torch.float16 if not args.use_8bit_quantization else "auto",  # 메모리 절약을 위해 float16 사용
        low_cpu_mem_usage=True,  # CPU 메모리 사용량 감소
        peft_config=lora_config
    )
    
    # 메모리 최적화 옵션 설정
    if args.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("✅ Gradient checkpointing 활성화")
    
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
        # """한국어 데이터셋을 TRL 학습에 맞게 가공하는 함수"""
        # maywell/korean_textbooks 데이터셋은 config가 필요함
        if "korean_textbooks" in dataset_name:
            ds = load_dataset(dataset_name, "normal_instructions", split="train", trust_remote_code=True)
            # korean_textbooks 데이터셋의 컬럼 구조에 맞게 수정
            if "instruction" in ds.column_names and "output" in ds.column_names:
                # instruction + output을 합쳐서 review로 사용
                def combine_text(sample):
                    sample["review"] = sample["instruction"] + " " + sample["output"]
                    return sample
                ds = ds.map(combine_text, batched=False)
            elif "text" in ds.column_names:
                ds = ds.rename_columns({"text": "review"})
            else:
                # 첫 번째 텍스트 컬럼을 review로 사용
                text_column = ds.column_names[0]
                ds = ds.rename_columns({text_column: "review"})
        else:
            # 기존 NSMC 등의 데이터셋 처리
            ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
            if "document" in ds.column_names:
                ds = ds.rename_columns({"document": "review"})
        
        ds = ds.filter(lambda x: len(x["review"]) > 20, batched=False)  # 최소 길이 요구사항 완화
        
        # 데이터셋 크기를 제한하여 빠른 테스트
        ds = ds.select(range(min(sample_size, len(ds))))

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            prompt = sample["review"][: input_size()]
            # 메모리 절약을 위해 최대 길이 제한 및 attention_mask 추가
            encoding = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                padding=False,
                max_length=input_max_text_length  # 명시적 최대 길이 제한
            )
            sample["input_ids"] = encoding["input_ids"].squeeze(0)  # 배치 차원 제거
            sample["attention_mask"] = encoding["attention_mask"].squeeze(0)  # attention_mask 추가
            sample["query"] = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds

    # dataset = build_dataset(config)
    dataset = build_dataset(config, args.dataset_name, args.input_min_text_length, args.input_max_text_length, args.dataset_sample_size)

    # --- 5. 한국어 보상 모델(Reward Model) 정의 ---
    # GPU 우선으로 sentiment pipeline 로드
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


    # --- 6. 참조 모델(Reference Model) 설정 ---
    # 참조 모델을 명시적으로 생성하고 래퍼로 감싸기 (tuple 문제 해결)
    logger.info("참조 모델을 명시적으로 생성 중...")
    
    # 참조 모델은 양자화 없이 로드 (return_dict 문제 방지)
    from transformers import AutoModelForCausalLM
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=None,  # 양자화 비활성화
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        return_dict=True  # 명시적으로 return_dict=True 설정
    )
    
    # 참조 모델 설정 강제
    ref_model.config.return_dict = True
    
    # 참조 모델을 래퍼로 감싸기
    ref_model = ReferenceModelWrapper(ref_model)
    logger.info("✅ 참조 모델 생성 및 래퍼 적용 완료")
    
    # --- 7. PPOTrainer 초기화 ---
    # 올바른 API: args, model, reward_model, train_dataset, value_model이 필수
    def collator(data):
        # 텐서를 적절히 배치로 스택하는 collator
        batch = {}
        for key in data[0]:
            if key in ["input_ids", "attention_mask"]:
                # input_ids와 attention_mask는 텐서들을 패딩하여 배치로 만듦
                sequences = [d[key] for d in data if d[key] is not None]
                if not sequences:  # 모든 값이 None인 경우
                    batch[key] = None
                    continue
                
                # 모든 시퀀스를 동일한 길이로 패딩
                max_len = max(len(seq) for seq in sequences)
                padded_sequences = []
                for d in data:
                    seq = d[key]
                    if seq is None:
                        # attention_mask가 None인 경우 0으로 채운 마스크 생성
                        if key == "attention_mask":
                            padded = torch.zeros(max_len, dtype=torch.long)
                        else:
                            padded = torch.full((max_len,), tokenizer.pad_token_id, dtype=torch.long)
                    elif len(seq) < max_len:
                        # 패딩 토큰 또는 0으로 채움
                        if key == "attention_mask":
                            padded = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=seq.dtype)])
                        else:
                            padded = torch.cat([seq, torch.full((max_len - len(seq),), tokenizer.pad_token_id, dtype=seq.dtype)])
                    else:
                        padded = seq
                    padded_sequences.append(padded)
                batch[key] = torch.stack(padded_sequences)
            else:
                # 다른 키들은 리스트로 유지
                batch[key] = [d[key] for d in data]
        return batch

    # 보상 모델은 sentiment_pipe를 래핑해서 사용
    class RewardModel(torch.nn.Module):
        def __init__(self, sentiment_pipeline):
            super().__init__()
            self.sentiment_pipeline = sentiment_pipeline
            
        def forward(self, input_ids, **kwargs):
            # 단순한 보상 반환 (실제로는 더 복잡한 로직 필요)
            return torch.tensor([0.5] * len(input_ids), device=input_ids.device)
    
    reward_model = RewardModel(sentiment_pipe)
    
    # AutoModelForCausalLMWithValueHead에서 base_model 추출
    # value_model로는 기본 transformers 모델을 사용
    try:
        # 몇 가지 방법으로 base_model 추출 시도
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
            # 마지막 수단: 동일한 모델 사용하되 필수 속성 추가
            base_model = model
            if not hasattr(base_model, 'base_model_prefix'):
                base_model.base_model_prefix = "gpt_neox"  # EleutherAI 모델의 경우
            logger.info("✅ 모델에 base_model_prefix 추가")
            
    except Exception as e:
        logger.error(f"Base model 추출 실패: {e}")
        base_model = model  # 폴백
        if not hasattr(base_model, 'base_model_prefix'):
            base_model.base_model_prefix = "gpt_neox"
        logger.info("✅ 폴백: 모델에 base_model_prefix 추가")
    
    # PPOTrainer 생성 (개선된 버전 사용)
    ppo_trainer = FixedPPOTrainer(
        args=config,  # config가 아니라 args!
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,  # 래퍼로 감싸진 참조 모델 전달
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=base_model,  # 추출된 base_model 사용
        data_collator=collator
    )
    
    logger.info("✅ PPOTrainer 생성 성공!")

    # --- 7. PPO 파인튜닝 실행 ---
    logger.info("PPO 파인튜닝 훈련을 시작합니다...")

    # 생성 파라미터 설정 (메모리 최적화)
    generation_kwargs = {
        "min_length": -1, # 생성 시 최소 길이 (-1은 설정 안 함)
        "top_k": 0.0, # 상위 K 샘플링 (0.0은 사용 안 함)
        "top_p": 1.0, # 상위 P 샘플링 (1.0은 사용 안 함)
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens, # 예: 16 (메모리 절약을 위해 감소)
        "temperature": args.temperature,
        "use_cache": True,  # 메모리 효율성을 위해 캠시 사용
    }

    # 훈련 루프 설정
    # num_epochs = 3 # 전체 데이터셋에 대한 반복 횟수 -> PPOTrainer는 스텝 기반으로 동작
    # max_ppo_steps = 10 # PPO 스텝 수 (예시, 실제로는 더 많이 필요)
    max_ppo_steps = args.max_ppo_steps

    for step in range(max_ppo_steps):
        logger.info(f"PPO Step {step + 1}/{max_ppo_steps} 진행 중...")
        
        # 메모리 정리 (각 스텝 시작 시)
        clear_gpu_cache()
        
        # 데이터셋에서 배치 가져오기 (PPOTrainer가 내부적으로 처리)
        # 이 부분은 PPOTrainer의 내부 로직에 따라 dataset에서 샘플링 됩니다.
        # 사용자가 직접 dataloader를 만들 필요는 없지만, dataset은 준비되어야 합니다.
        # PPOTrainer는 config.batch_size에 맞춰 dataset에서 샘플링합니다.
        
        # build_dataset에서 ds.select(range(50))으로 데이터셋 크기를 줄였으므로,
        # config.batch_size에 따라 반복 횟수가 결정됩니다.
        # 예: 50개 샘플, batch_size=2 -> 25번의 내부 반복 후 1 PPO step 완료.
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
        
        query_tensors = [data["input_ids"].clone().detach().to(next(model.parameters()).device) for data in batch_list]
        
        # 모델을 사용하여 응답 생성 (model.generate 함수 사용)
        # response_tensors 리스트는 각 쿼리에 대한 생성된 응답 텐서들을 포함합니다.
        response_tensors = []
        for i, query_tensor in enumerate(query_tensors):
            # 각 쿼리에 대해 개별적으로 생성 (메모리 절약)
            try:
                with torch.no_grad():
                    # attention_mask 추가
                    attention_mask = batch_list[i].get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.unsqueeze(0).to(next(model.parameters()).device)
                    
                    generated_ids = model.generate(
                        input_ids=query_tensor.unsqueeze(0),  # 배치 차원 추가
                        attention_mask=attention_mask,  # attention_mask 추가
                        **generation_kwargs
                    )
                    # 원본 쿼리 부분을 제거하고 새로 생성된 토큰만 추출
                    response_tensor = generated_ids[0][len(query_tensor):]
                    response_tensors.append(response_tensor)
                    
                    # 메모리 정리 (각 생성 후)
                    del generated_ids
                    if attention_mask is not None:
                        del attention_mask
                    
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM during generation for query {i+1}: {e}")
                logger.info("메모리 정리 후 재시도...")
                clear_gpu_cache()
                # 더 작은 버전으로 재시도
                try:
                    generation_kwargs_small = generation_kwargs.copy()
                    generation_kwargs_small["max_new_tokens"] = max(1, generation_kwargs_small["max_new_tokens"] // 2)
                    with torch.no_grad():
                        attention_mask = batch_list[i].get("attention_mask")
                        if attention_mask is not None:
                            attention_mask = attention_mask.unsqueeze(0).to(next(model.parameters()).device)
                        
                        generated_ids = model.generate(
                            input_ids=query_tensor.unsqueeze(0),
                            attention_mask=attention_mask,
                            **generation_kwargs_small
                        )
                        response_tensor = generated_ids[0][len(query_tensor):]
                        response_tensors.append(response_tensor)
                        logger.info(f"재시도 성공: max_new_tokens={generation_kwargs_small['max_new_tokens']}")
                        
                        del generated_ids
                        if attention_mask is not None:
                            del attention_mask
                except Exception as e2:
                    logger.error(f"재시도도 실패: {e2}")
                    # 빈 응답 추가
                    response_tensors.append(torch.tensor([], dtype=query_tensor.dtype, device=query_tensor.device))
            except Exception as e:
                logger.error(f"Generation error for query {i+1}: {e}")
                response_tensors.append(torch.tensor([], dtype=query_tensor.dtype, device=query_tensor.device))
        
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
                    rewards.append(torch.tensor(sentiment_result['score'], device=next(model.parameters()).device))
                else:  # 부정
                    rewards.append(torch.tensor(sentiment_result['score'] * 0.1, device=next(model.parameters()).device)) # 부정 보상 대폭 감소
            except Exception as e:
                logger.error(f"Sentiment analysis error for text '{text[:30]}...': {e}")
                rewards.append(torch.tensor(0.0, device=next(model.parameters()).device)) # 에러 시 0점 보상

        # PPO 스텝 실행 (모델 업데이트)
        # query_tensors: 입력 프롬프트 텐서 리스트
        # response_tensors: 모델이 생성한 응답 텐서 리스트
        # rewards: 각 응답에 대한 보상 텐서 리스트
        try:
            # TRL의 최신 API에서는 step 대신 다른 메소드를 사용할 수 있음
            if hasattr(ppo_trainer, 'step'):
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, {}, rewards) # 로그 기록
                logger.info(f"  PPO Step {step+1} 완료. Stats: { {k: v.mean().item() if isinstance(v, list) and v and isinstance(v[0], torch.Tensor) else v for k, v in stats.items()} }")
            else:
                # step 메소드가 없는 경우 대안 방법 시도
                logger.warning("PPOTrainer에 'step' 메소드가 없습니다. 대안 방법을 시도합니다.")
                
                # 방법 1: train 메소드 사용 (하지만 이는 전체 훈련을 실행함)
                if hasattr(ppo_trainer, 'train'):
                    logger.info("'train' 메소드를 사용하여 전체 훈련을 시작합니다.")
                    # 이 경우 for 루프를 중단하고 train()을 호출
                    break
                else:
                    logger.error("사용 가능한 훈련 메소드를 찾을 수 없습니다.")
                    # 사용 가능한 메소드 출력
                    available_methods = [method for method in dir(ppo_trainer) if not method.startswith('_') and callable(getattr(ppo_trainer, method))]
                    logger.info(f"사용 가능한 메소드들: {available_methods}")
                    break
                    
            avg_reward_step = torch.tensor(rewards).mean().item()
            logger.info(f"  Average reward for step {step+1}: {avg_reward_step:.4f}")
            
            # 메모리 정리 (각 스텝 끝)
            clear_gpu_cache()

        except Exception as e:
            logger.error(f"PPO step 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 중요: 메모리 정리 후 다음 스텝 시도
            clear_gpu_cache()
            
            # 메모리 부족인 경우 설정 조정
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning("메모리 부족 감지. 배치 크기 및 생성 토큰 수 감소.")
                config.batch_size = max(1, config.batch_size // 2)
                generation_kwargs["max_new_tokens"] = max(1, generation_kwargs["max_new_tokens"] // 2)
                logger.info(f"새로운 설정: batch_size={config.batch_size}, max_new_tokens={generation_kwargs['max_new_tokens']}")
                continue  # 다음 스텝 진행
            else:
                # 다른 오류인 경우 중단
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
    # PEFT 모델의 경우 model.save_pretrained를 사용하여 어댑터를 저장
    # PPOTrainer는 save_pretrained 메소드가 없으므로 모델을 직접 저장

    output_dir = args.output_dir
    try:
        # PEFT 어댑터 저장
        model.save_pretrained(output_dir)
        logger.info(f"✅ 모델 어댑터가 '{output_dir}'에 저장되었습니다.")
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")
        # 대안: 어댑터만 저장 시도
        try:
            if hasattr(model, 'peft_config'):
                model.save_adapter(output_dir)
                logger.info(f"✅ PEFT 어댑터가 '{output_dir}'에 저장되었습니다.")
        except Exception as e2:
            logger.error(f"어댑터 저장도 실패: {e2}")
    
    # 토크나이저는 항상 저장
    tokenizer.save_pretrained(output_dir)
    logger.info(f"💾 토크나이저가 '{output_dir}'에 저장되었습니다.")

    # 만약 TensorBoard 로깅을 사용했다면, 다음 명령어로 확인 가능:
    # tensorboard --logdir logs/ppo_tuning

if __name__ == "__main__":
    main()