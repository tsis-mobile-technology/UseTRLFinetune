import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Quick test to see the output structure of the base model
model_name = "EleutherAI/polyglot-ko-1.3b"

print("Testing reference model output structure...")

try:
    # Load a minimal version without quantization for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Try without quantization first
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",  # Force CPU to avoid GPU memory issues
        torch_dtype=torch.float16  # Use float16 to save memory
    )
    
    # Test input
    test_input = tokenizer("안녕하세요", return_tensors="pt", padding=True)
    
    # Test output
    with torch.no_grad():
        output = ref_model(**test_input)
    
    print(f"Output type: {type(output)}")
    print(f"Output attributes: {dir(output)}")
    
    if hasattr(output, 'logits'):
        print(f"✅ Has logits attribute. Shape: {output.logits.shape}")
    else:
        print("❌ No logits attribute!")
        
    if hasattr(output, 'last_hidden_state'):
        print(f"Has last_hidden_state. Shape: {output.last_hidden_state.shape}")
        
    # Check what the output actually is
    print(f"Output content: {output}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()