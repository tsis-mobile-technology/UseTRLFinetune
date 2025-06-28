#!/usr/bin/env python3

import subprocess
import sys
import os

# Set working directory
os.chdir('/media/proidea/hdd/Programming/UseTRLFinetune')

# Set environment variable
env = os.environ.copy()
env['PYTHONPATH'] = '/media/proidea/hdd/Programming/UseTRLFinetune:' + env.get('PYTHONPATH', '')

print("Running train.py with corrected reference model...")

# Run the command with very small parameters for quick testing
cmd = [
    '/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python',
    'train.py',
    '--model_name', 'EleutherAI/polyglot-ko-1.3b',
    '--dataset_name', 'maywell/korean_textbooks',
    '--output_dir', 'my_ref_fixed_test',
    '--batch_size', '1',
    '--mini_batch_size', '1',
    '--max_ppo_steps', '1',
    '--dataset_sample_size', '5',
    '--max_new_tokens', '8',
    '--learning_rate', '1e-5'
]

try:
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
    # Check for specific error patterns
    if "'tuple' object has no attribute 'logits'" in result.stderr:
        print("\n❌ Still getting the tuple/logits AttributeError!")
    elif "AttributeError" in result.stderr:
        print("\n⚠️  Different AttributeError detected")
    elif result.returncode == 0:
        print("\n✅ Script completed successfully!")
    else:
        print(f"\n⚠️  Script failed with return code {result.returncode}")
        
except subprocess.TimeoutExpired:
    print("Process timed out after 10 minutes")
except Exception as e:
    print(f"Error running command: {e}")