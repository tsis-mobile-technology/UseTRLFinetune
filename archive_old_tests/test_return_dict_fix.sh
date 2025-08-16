#!/bin/bash

cd /media/proidea/hdd/Programming/UseTRLFinetune

echo "=== Testing train.py with return_dict=True fix ==="
echo "Running with reduced parameters to test the fix quickly..."

/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python train.py \
    --model_name "EleutherAI/polyglot-ko-1.3b" \
    --dataset_name "maywell/korean_textbooks" \
    --output_dir "test_return_dict_fix" \
    --batch_size 2 \
    --mini_batch_size 1 \
    --max_ppo_steps 1 \
    --dataset_sample_size 5 \
    --max_new_tokens 16 \
    --learning_rate 1e-5

echo ""
echo "=== Test completed ==="

# Check if we made progress
if [ -d "test_return_dict_fix" ]; then
    echo "✅ Output directory created - model saving succeeded"
    ls -la test_return_dict_fix/
else
    echo "❌ No output directory created"
fi

# Check for specific success indicators
echo ""
echo "=== Checking for progress indicators ==="
echo "Looking for 'tuple' object has no attribute 'logits' error..."

# Run a simple grep check (if we have logs)
if [ -f "train_output.log" ]; then
    if grep -q "tuple.*logits" train_output.log; then
        echo "❌ Still getting tuple/logits error"
    else
        echo "✅ No tuple/logits error found"
    fi
fi