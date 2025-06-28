#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/media/proidea/hdd/Programming/UseTRLFinetune')
os.chdir('/media/proidea/hdd/Programming/UseTRLFinetune')

print("=== Environment and Version Check ===")

try:
    import trl
    print(f"TRL version: {trl.__version__}")
except:
    print("TRL version: Unable to determine")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except:
    print("Transformers version: Unable to determine")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except:
    print("PyTorch version: Unable to determine")

# Check if return_dict is actually being respected
print("\n=== Testing return_dict behavior ===")

try:
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import AutoModelForCausalLMWithValueHead
    
    model_name = "EleutherAI/polyglot-ko-1.3b"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model (this will take a moment)...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    print(f"Model config return_dict before setting: {getattr(model.config, 'return_dict', 'Not set')}")
    
    # Set return_dict
    model.config.return_dict = True
    
    print(f"Model config return_dict after setting: {model.config.return_dict}")
    
    # Test output
    test_input = tokenizer("안녕하세요", return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    test_input = {k: v.to(device) for k, v in test_input.items()}
    
    with torch.no_grad():
        output = model(**test_input)
    
    print(f"Output type: {type(output)}")
    print(f"Is tuple: {isinstance(output, tuple)}")
    print(f"Has logits: {hasattr(output, 'logits')}")
    
    if hasattr(output, 'logits'):
        print(f"✅ Model correctly returns ModelOutput with logits")
    else:
        print(f"❌ Model returns tuple or object without logits")
        print(f"Output content: {output}")

except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Environment check completed ===")