import subprocess
import sys
import os

# Set working directory
os.chdir('/media/proidea/hdd/Programming/UseTRLFinetune')

# Set environment variable
env = os.environ.copy()
env['PYTHONPATH'] = '/media/proidea/hdd/Programming/UseTRLFinetune:' + env.get('PYTHONPATH', '')

print("Running train.py with reference model fix...")

# Run the command
cmd = [
    '/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python',
    'train.py',
    '--model_name', 'EleutherAI/polyglot-ko-1.3b',
    '--dataset_name', 'maywell/korean_textbooks',
    '--output_dir', 'my_ref_model_test',
    '--batch_size', '2',
    '--mini_batch_size', '1',
    '--max_ppo_steps', '1',
    '--dataset_sample_size', '10',
    '--max_new_tokens', '16',
    '--learning_rate', '1e-5'
]

try:
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
except subprocess.TimeoutExpired:
    print("Process timed out after 10 minutes")
except Exception as e:
    print(f"Error running command: {e}")