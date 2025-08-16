import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_ppo_models():
    """Debug what's happening with the reference model in PPOTrainer"""
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create test input
        test_text = "안녕하세요. 오늘은"
        test_input = tokenizer(test_text, return_tensors="pt", padding=True)
        print(f"Test input shape: {test_input['input_ids'].shape}")
        
        # 1. Test without quantization first (to isolate the issue)
        print("\n=== Testing without quantization ===")
        
        # Load reference model (simple)
        ref_model_simple = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float16
        )
        
        with torch.no_grad():
            ref_output = ref_model_simple(**test_input)
            print(f"Simple ref model output type: {type(ref_output)}")
            print(f"Has logits: {hasattr(ref_output, 'logits')}")
            if hasattr(ref_output, 'logits'):
                print(f"Logits shape: {ref_output.logits.shape}")
        
        # 2. Test with quantization (current setup)
        print("\n=== Testing with quantization ===")
        
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
            print(f"Quantized ref model output type: {type(ref_output)}")
            print(f"Has logits: {hasattr(ref_output, 'logits')}")
            if hasattr(ref_output, 'logits'):
                print(f"Logits shape: {ref_output.logits.shape}")
            else:
                print(f"Output content: {ref_output}")
        
        # 3. Test the main model with LoRA
        print("\n=== Testing main model with LoRA ===")
        
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
            print(f"Main model output type: {type(main_output)}")
            print(f"Has logits: {hasattr(main_output, 'logits')}")
            if hasattr(main_output, 'logits'):
                print(f"Logits shape: {main_output.logits.shape}")
        
        # 4. Check if PPOTrainer expects something specific
        print("\n=== Checking PPOTrainer expectations ===")
        
        # Try to see what PPOTrainer does internally
        from trl.trainer.ppo_trainer import PPOTrainer
        import inspect
        
        # Look at the PPOTrainer source code for the problematic method
        if hasattr(PPOTrainer, '_get_batch_logps'):
            signature = inspect.signature(PPOTrainer._get_batch_logps)
            print(f"_get_batch_logps signature: {signature}")
        
        # Check if there are any specific requirements for the ref_model
        print("PPOTrainer.__init__ signature:")
        init_signature = inspect.signature(PPOTrainer.__init__)
        print(init_signature)
        
        print("\n=== Testing actual PPOTrainer creation ===")
        
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
            print("✅ PPOTrainer created successfully with quantized ref_model!")
            
        except Exception as e:
            print(f"❌ PPOTrainer creation failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Overall error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ppo_models()