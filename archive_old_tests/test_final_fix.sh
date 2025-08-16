#!/bin/bash

cd /media/proidea/hdd/Programming/UseTRLFinetune

echo "=== Testing train.py with create_reference_model fix ==="
echo "Running with the original parameters but reduced for testing..."

/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python train.py \
    --model_name "EleutherAI/polyglot-ko-1.3b" \
    --dataset_name "maywell/korean_textbooks" \
    --output_dir "test_create_ref_model_final" \
    --batch_size 2 \
    --mini_batch_size 1 \
    --max_ppo_steps 1 \
    --dataset_sample_size 5 \
    --max_new_tokens 16 \
    --learning_rate 1e-5

echo ""
echo "=== Test completed ==="

# Check if we made progress
if [ -d "test_create_ref_model_final" ]; then
    echo "✅ Output directory created - model saving succeeded"
    ls -la test_create_ref_model_final/
else
    echo "❌ No output directory created"
fi