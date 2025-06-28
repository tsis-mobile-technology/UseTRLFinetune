#!/bin/bash
cd /media/proidea/hdd/Programming/UseTRLFinetune
export PYTHONPATH="/media/proidea/hdd/Programming/UseTRLFinetune:$PYTHONPATH"

echo "Running train.py with reference model fix..."
/media/proidea/hdd/Programming/UseTRLFinetune/korean-llm-env/bin/python train.py \
    --model_name "EleutherAI/polyglot-ko-1.3b" \
    --dataset_name "maywell/korean_textbooks" \
    --output_dir "my_ref_model_test" \
    --batch_size 2 \
    --mini_batch_size 1 \
    --max_ppo_steps 1 \
    --dataset_sample_size 10 \
    --max_new_tokens 16 \
    --learning_rate 1e-5