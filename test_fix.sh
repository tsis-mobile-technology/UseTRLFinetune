#!/bin/bash

cd /media/proidea/hdd/Programming/UseTRLFinetune

echo "=== Testing the fixed train.py with proper reference model ==="
echo "Running with minimal parameters to test the fix quickly..."

/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python train.py \
    --model_name "EleutherAI/polyglot-ko-1.3b" \
    --dataset_name "maywell/korean_textbooks" \
    --output_dir "test_ref_fix" \
    --batch_size 1 \
    --mini_batch_size 1 \
    --max_ppo_steps 1 \
    --dataset_sample_size 3 \
    --max_new_tokens 8 \
    --learning_rate 1e-5

echo "=== Test completed ==="