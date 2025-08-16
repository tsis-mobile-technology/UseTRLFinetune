#!/bin/bash

cd /media/proidea/hdd/Programming/UseTRLFinetune

echo "Testing reference model wrapper..."
/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python test_wrapper.py

echo ""
echo "If wrapper test passes, running training with wrapper..."
echo "Press Ctrl+C to stop if needed"
echo ""

# Run the training script with wrapper
/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python train_with_wrapper.py \
  --model_name "EleutherAI/polyglot-ko-1.3b" \
  --dataset_name "maywell/korean_textbooks" \
  --output_dir "my_korean_ppo_wrapper_model" \
  --max_ppo_steps 3 \
  --batch_size 2 \
  --mini_batch_size 1 \
  --dataset_sample_size 50