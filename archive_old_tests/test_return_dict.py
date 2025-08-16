import torch
import logging
from transformers import AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_return_dict_fix():
    """Test if setting return_dict=True fixes the tuple issue"""
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    try:
        # 1. Create tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 2. Setup configurations
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
        
        # 3. Load main model with LoRA
        logger.info("Loading main model with LoRA...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            peft_config=lora_config
        )
        
        # Set return_dict=True
        model.config.return_dict = True
        
        if not hasattr(model, 'generation_config'):
            model.generation_config = GenerationConfig.from_pretrained(model_name)
        
        # 4. Load reference model (separate load, no LoRA)
        logger.info("Loading reference model...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            # No peft_config
        )
        
        # Set return_dict=True for reference model
        ref_model.config.return_dict = True
        ref_model.config.pad_token_id = tokenizer.eos_token_id
        
        if not hasattr(ref_model, 'generation_config'):
            ref_model.generation_config = GenerationConfig.from_pretrained(model_name)
        
        # Set reference model to eval mode
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
            
        logger.info("✅ Both models loaded successfully")
        
        # 5. Test model outputs - this is the crucial test
        logger.info("Testing model outputs...")
        test_input = tokenizer("안녕하세요", return_tensors="pt", padding=True)
        
        # Move to same device as models
        device = next(model.parameters()).device
        test_input = {k: v.to(device) for k, v in test_input.items()}
        
        with torch.no_grad():
            # Test main model
            main_output = model(**test_input)
            logger.info(f"Main model output type: {type(main_output)}")
            logger.info(f"Main model output is tuple: {isinstance(main_output, tuple)}")
            if hasattr(main_output, 'logits'):
                logger.info(f"✅ Main model has logits attribute. Shape: {main_output.logits.shape}")
            else:
                logger.error(f"❌ Main model output has no logits attribute!")
                logger.info(f"Main model output content: {main_output}")
                return False
            
            # Test reference model - this is where the error was happening
            ref_output = ref_model(**test_input)
            logger.info(f"Reference model output type: {type(ref_output)}")
            logger.info(f"Reference model output is tuple: {isinstance(ref_output, tuple)}")
            if hasattr(ref_output, 'logits'):
                logger.info(f"✅ Reference model has logits attribute. Shape: {ref_output.logits.shape}")
            else:
                logger.error(f"❌ Reference model output has no logits attribute!")
                logger.info(f"Reference model output content: {ref_output}")
                return False
        
        logger.info("✅ Both models produce ModelOutput objects with logits")
        
        # 6. Test PPOTrainer creation
        logger.info("Testing PPOTrainer creation...")
        
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=1,
            mini_batch_size=1
        )
        
        # Simple reward model
        class DummyRewardModel(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return torch.tensor([0.5] * len(input_ids), device=input_ids.device)
        
        reward_model = DummyRewardModel()
        
        # Extract base model for value_model
        if hasattr(model, 'pretrained_model'):
            base_model = model.pretrained_model
        elif hasattr(model, 'base_model'):
            base_model = model.base_model
        else:
            base_model = model
            
        # Create minimal dataset
        from datasets import Dataset
        minimal_data = {
            "input_ids": [test_input["input_ids"].squeeze()],
            "attention_mask": [test_input["attention_mask"].squeeze()],
            "query": ["안녕하세요"]
        }
        dataset = Dataset.from_dict(minimal_data)
        dataset.set_format(type="torch")
        
        # Simple collator
        def collator(data):
            return {
                "input_ids": torch.stack([d["input_ids"] for d in data]),
                "attention_mask": torch.stack([d["attention_mask"] for d in data]),
                "query": [d["query"] for d in data]
            }
        
        # Try to create PPOTrainer
        ppo_trainer = PPOTrainer(
            args=config,
            processing_class=tokenizer,
            model=model,
            ref_model=ref_model,  # Using separate reference model with return_dict=True
            reward_model=reward_model,
            train_dataset=dataset,
            value_model=base_model,
            data_collator=collator
        )
        
        logger.info("✅ PPOTrainer created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_return_dict_fix()
    if success:
        print("\n🎉 return_dict fix test PASSED!")
    else:
        print("\n💥 return_dict fix test FAILED!")