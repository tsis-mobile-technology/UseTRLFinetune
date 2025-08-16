#!/usr/bin/env python3
"""
통합 디버깅 도구
기존의 debug_로 시작하는 모든 스크립트들을 하나로 통합
"""

import os
import sys
import torch
import logging
import argparse
import inspect
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 프로젝트 경로
PROJECT_ROOT = "/media/proidea/hdd/Programming/UseTRLFinetune"

def debug_ppo_methods():
    """PPO 트레이너 메소드 확인 (기존 debug_ppo_methods.py)"""
    logger.info("=== PPO 트레이너 메소드 디버깅 ===")
    
    try:
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        
        model_name = "EleutherAI/polyglot-ko-1.3b"
        
        # 빠른 설정으로 작은 모델 로드
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=1,
            mini_batch_size=1
        )
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        
        logger.info("모델 로드 중...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            peft_config=lora_config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 더미 데이터셋
        class DummyDataset:
            def __init__(self):
                self.data = [{"input_ids": torch.tensor([1, 2, 3])}]
            
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                return self.data[0]
        
        dataset = DummyDataset()
        
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])
        
        # 더미 reward model
        class DummyRewardModel(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return torch.tensor([0.5] * len(input_ids), device=input_ids.device)
        
        reward_model = DummyRewardModel()
        
        # base model 추출
        if hasattr(model, 'pretrained_model'):
            base_model = model.pretrained_model
        elif hasattr(model, 'base_model'):
            base_model = model.base_model
        else:
            base_model = model
        
        logger.info("PPOTrainer 생성 중...")
        try:
            ppo_trainer = PPOTrainer(
                args=config,
                processing_class=tokenizer,
                model=model,
                ref_model=None,
                reward_model=reward_model,
                train_dataset=dataset,
                value_model=base_model,
                data_collator=collator
            )
            logger.info("✅ PPOTrainer 생성 성공!")
            
            logger.info("\n사용 가능한 메소드들:")
            methods = [method for method in dir(ppo_trainer) if not method.startswith('_')]
            for method in sorted(methods):
                logger.info(f"  - {method}")
            
            logger.info(f"\n'step' 메소드 존재 여부: {'step' in methods}")
            logger.info(f"'train' 메소드 존재 여부: {'train' in methods}")
            logger.info(f"'generate' 메소드 존재 여부: {'generate' in methods}")
            
            # 실제 속성과 타입 확인
            if hasattr(ppo_trainer, 'step'):
                logger.info(f"\nstep 메소드 타입: {type(ppo_trainer.step)}")
            else:
                logger.warning("\n❌ step 메소드가 존재하지 않습니다!")
            
            # 대안 메소드 확인
            if hasattr(ppo_trainer, 'train'):
                logger.info(f"train 메소드 타입: {type(ppo_trainer.train)}")
            if hasattr(ppo_trainer, 'update'):
                logger.info(f"update 메소드 타입: {type(ppo_trainer.update)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ PPOTrainer 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except ImportError as e:
        logger.error(f"❌ 필요한 모듈을 가져올 수 없습니다: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_reference_model():
    """참조 모델 디버깅 (기존 debug_ref_model.py)"""
    logger.info("=== 참조 모델 디버깅 ===")
    
    try:
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        from peft import LoraConfig
        
        model_name = "EleutherAI/polyglot-ko-1.3b"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create test input
        test_text = "안녕하세요. 오늘은"
        test_input = tokenizer(test_text, return_tensors="pt", padding=True)
        logger.info(f"Test input shape: {test_input['input_ids'].shape}")
        
        # 1. Test without quantization first (to isolate the issue)
        logger.info("\n=== Testing without quantization ===")
        
        # Load reference model (simple)
        ref_model_simple = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float16
        )
        
        with torch.no_grad():
            ref_output = ref_model_simple(**test_input)
            logger.info(f"Simple ref model output type: {type(ref_output)}")
            logger.info(f"Has logits: {hasattr(ref_output, 'logits')}")
            if hasattr(ref_output, 'logits'):
                logger.info(f"Logits shape: {ref_output.logits.shape}")
        
        # 2. Test with quantization (current setup)
        logger.info("\n=== Testing with quantization ===")
        
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        ref_model_quant = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # Move test input to same device
        device = next(ref_model_quant.parameters()).device
        test_input_gpu = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            ref_output = ref_model_quant(**test_input_gpu)
            logger.info(f"Quantized ref model output type: {type(ref_output)}")
            logger.info(f"Has logits: {hasattr(ref_output, 'logits')}")
            if hasattr(ref_output, 'logits'):
                logger.info(f"Logits shape: {ref_output.logits.shape}")
            else:
                logger.info(f"Output content: {ref_output}")
        
        # 3. Test the main model with LoRA
        logger.info("\n=== Testing main model with LoRA ===")
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        main_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            peft_config=lora_config
        )
        
        with torch.no_grad():
            main_output = main_model(**test_input_gpu)
            logger.info(f"Main model output type: {type(main_output)}")
            logger.info(f"Has logits: {hasattr(main_output, 'logits')}")
            if hasattr(main_output, 'logits'):
                logger.info(f"Logits shape: {main_output.logits.shape}")
        
        # 4. Check if PPOTrainer expects something specific
        logger.info("\n=== Checking PPOTrainer expectations ===")
        
        # Try to see what PPOTrainer does internally
        from trl.trainer.ppo_trainer import PPOTrainer
        
        # Look at the PPOTrainer source code for the problematic method
        if hasattr(PPOTrainer, '_get_batch_logps'):
            signature = inspect.signature(PPOTrainer._get_batch_logps)
            logger.info(f"_get_batch_logps signature: {signature}")
        
        # Check if there are any specific requirements for the ref_model
        logger.info("PPOTrainer.__init__ signature:")
        init_signature = inspect.signature(PPOTrainer.__init__)
        logger.info(str(init_signature))
        
        logger.info("\n=== Testing actual PPOTrainer creation ===")
        
        # Try the actual PPOTrainer creation (minimal)
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=1,
            mini_batch_size=1
        )
        
        class DummyRewardModel(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return torch.tensor([0.5] * len(input_ids), device=input_ids.device)
        
        reward_model = DummyRewardModel()
        
        # Try with the quantized reference model
        try:
            from datasets import Dataset
            minimal_data = {
                "input_ids": [test_input_gpu["input_ids"].squeeze()],
                "attention_mask": [test_input_gpu["attention_mask"].squeeze()],
                "query": [test_text]
            }
            dataset = Dataset.from_dict(minimal_data)
            dataset.set_format(type="torch")
            
            ppo_trainer = PPOTrainer(
                args=config,
                processing_class=tokenizer,
                model=main_model,
                ref_model=ref_model_quant,  # Using quantized reference model
                reward_model=reward_model,
                train_dataset=dataset,
                value_model=main_model.pretrained_model if hasattr(main_model, 'pretrained_model') else main_model,
                data_collator=lambda x: x[0]
            )
            logger.info("✅ PPOTrainer created successfully with quantized ref_model!")
            return True
            
        except Exception as e:
            logger.error(f"❌ PPOTrainer creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        logger.error(f"Overall error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_reference_model_output():
    """참조 모델 출력 형태 테스트 (기존 debug_ref_model_output.py)"""
    logger.info("=== 참조 모델 출력 형태 테스트 ===")
    
    try:
        from trl import AutoModelForCausalLMWithValueHead
        from trl.models import create_reference_model
        
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
        
        return True
        
    except Exception as e:
        logger.error(f"참조 모델 출력 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_memory_usage():
    """메모리 사용량 분석"""
    logger.info("=== 메모리 사용량 분석 ===")
    
    try:
        import psutil
        import GPUtil
        
        # CPU 메모리
        memory = psutil.virtual_memory()
        logger.info(f"시스템 메모리: {memory.total / 1024**3:.1f} GB")
        logger.info(f"사용 중: {memory.used / 1024**3:.1f} GB ({memory.percent:.1f}%)")
        logger.info(f"사용 가능: {memory.available / 1024**3:.1f} GB")
        
        # GPU 메모리
        if torch.cuda.is_available():
            logger.info(f"\nGPU 정보:")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                
                logger.info(f"GPU {i} ({device_name}):")
                logger.info(f"  총 메모리: {memory_total:.1f} GB")
                logger.info(f"  할당됨: {memory_allocated:.1f} GB")
                logger.info(f"  예약됨: {memory_reserved:.1f} GB")
                logger.info(f"  사용률: {(memory_reserved/memory_total)*100:.1f}%")
        else:
            logger.warning("CUDA를 사용할 수 없어 GPU 메모리 정보를 가져올 수 없습니다.")
        
        return True
        
    except ImportError:
        logger.warning("psutil 또는 GPUtil 모듈이 없어 메모리 분석을 건너뜁니다.")
        return False
    except Exception as e:
        logger.error(f"메모리 분석 실패: {e}")
        return False

def check_package_versions():
    """패키지 버전 확인"""
    logger.info("=== 패키지 버전 확인 ===")
    
    packages_to_check = [
        "torch", "transformers", "trl", "peft", "bitsandbytes", 
        "datasets", "accelerate", "tokenizers"
    ]
    
    for package in packages_to_check:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            logger.info(f"{package}: {version}")
        except ImportError:
            logger.warning(f"{package}: Not installed")
        except Exception as e:
            logger.error(f"{package}: Error - {e}")
    
    # CUDA 정보
    if torch.cuda.is_available():
        logger.info(f"\nCUDA 정보:")
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    return True

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 디버깅 도구")
    parser.add_argument(
        "command",
        choices=[
            "ppo-methods", "ref-model", "ref-output", "memory", "versions", "all", "list"
        ],
        help="실행할 디버깅 명령"
    )
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("\n=== 사용 가능한 디버깅 명령어 ===")
        commands = {
            "ppo-methods": "PPO 트레이너 메소드 확인",
            "ref-model": "참조 모델 디버깅",
            "ref-output": "참조 모델 출력 형태 테스트",
            "memory": "메모리 사용량 분석",
            "versions": "패키지 버전 확인",
            "all": "모든 디버깅 테스트 실행",
            "list": "이 도움말 출력"
        }
        
        for cmd, desc in commands.items():
            print(f"  {cmd:12} : {desc}")
        
        print(f"\n사용법: python {sys.argv[0]} <command>")
        return
    
    logger.info("=== 통합 디버깅 도구 ===")
    
    # 작업 디렉토리 확인 및 변경
    if os.path.exists(PROJECT_ROOT):
        os.chdir(PROJECT_ROOT)
    
    success = True
    
    if args.command == "ppo-methods":
        success = debug_ppo_methods()
        
    elif args.command == "ref-model":
        success = debug_reference_model()
        
    elif args.command == "ref-output":
        success = debug_reference_model_output()
        
    elif args.command == "memory":
        success = analyze_memory_usage()
        
    elif args.command == "versions":
        success = check_package_versions()
        
    elif args.command == "all":
        logger.info("모든 디버깅 테스트를 실행합니다...")
        
        tests = [
            ("패키지 버전 확인", check_package_versions),
            ("메모리 사용량 분석", analyze_memory_usage),
            ("PPO 트레이너 메소드 확인", debug_ppo_methods),
            ("참조 모델 디버깅", debug_reference_model),
            ("참조 모델 출력 형태 테스트", debug_reference_model_output)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"테스트: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                results.append((test_name, result))
                
                if result:
                    logger.info(f"✅ {test_name} 완료")
                else:
                    logger.error(f"❌ {test_name} 실패")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} 오류: {e}")
                results.append((test_name, False))
        
        # 결과 요약
        logger.info(f"\n{'='*50}")
        logger.info("테스트 결과 요약")
        logger.info(f"{'='*50}")
        
        for test_name, result in results:
            status = "✅ 성공" if result else "❌ 실패"
            logger.info(f"{test_name}: {status}")
        
        success_count = sum(1 for _, result in results if result)
        total_count = len(results)
        logger.info(f"\n총 {total_count}개 테스트 중 {success_count}개 성공")
        
        success = success_count == total_count
    
    if success:
        logger.info(f"✅ '{args.command}' 디버깅이 성공적으로 완료되었습니다.")
    else:
        logger.error(f"❌ '{args.command}' 디버깅이 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()