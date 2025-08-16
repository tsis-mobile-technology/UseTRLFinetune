#!/usr/bin/env python3

import subprocess
import sys
import os

# Set working directory
os.chdir('/media/proidea/hdd/Programming/UseTRLFinetune')

# Set environment variable
env = os.environ.copy()
env['PYTHONPATH'] = '/media/proidea/hdd/Programming/UseTRLFinetune:' + env.get('PYTHONPATH', '')

print("=== Testing train.py with create_reference_model approach ===")
print("Running with minimal parameters to test the fix quickly...")

# Run the command with very small parameters for quick testing
cmd = [
    '/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python',
    'train.py',
    '--model_name', 'EleutherAI/polyglot-ko-1.3b',
    '--dataset_name', 'maywell/korean_textbooks',
    '--output_dir', 'test_create_ref_model',
    '--batch_size', '1',
    '--mini_batch_size', '1',
    '--max_ppo_steps', '1',
    '--dataset_sample_size', '3',
    '--max_new_tokens', '8',
    '--learning_rate', '1e-5'
]

try:
    print("Running command:", ' '.join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    
    print("\n=== STDOUT ===")
    print(result.stdout)
    print("\n=== STDERR ===")
    print(result.stderr)
    print(f"\n=== Return code: {result.returncode} ===")
    
    # Check for specific error patterns
    if "'tuple' object has no attribute 'logits'" in result.stderr:
        print("\n❌ Still getting the tuple/logits AttributeError!")
        return False
    elif "AttributeError" in result.stderr and "logits" in result.stderr:
        print("\n⚠️  Different logits-related AttributeError detected")
        return False
    elif result.returncode == 0:
        print("\n✅ Script completed successfully!")
        return True
    else:
        print(f"\n⚠️  Script failed with return code {result.returncode}")
        # Check if it's progress from our fix
        if "참조 모델 생성 완료" in result.stdout or "✅ 참조 모델 생성 완료" in result.stdout:
            print("✅ Reference model creation step passed!")
        if "✅ PPOTrainer 생성 성공" in result.stdout:
            print("✅ PPOTrainer creation step passed!")
        return False
        
except subprocess.TimeoutExpired:
    print("Process timed out after 15 minutes")
    return False
except Exception as e:
    print(f"Error running command: {e}")
    return False

if __name__ == "__main__":
    print("Running test directly...")
    sys.exit(0)  # Just run the subprocess test above