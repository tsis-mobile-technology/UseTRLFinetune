#!/usr/bin/env python3
import argparse

def test_argparse():
    parser = argparse.ArgumentParser(description="PPO 파인튜닝 스크립트")
    parser.add_argument("--model_name", type=str, default="EleutherAI/polyglot-ko-1.3b")
    parser.add_argument("--dataset_name", type=str, default="nsmc")
    parser.add_argument("--output_dir", type=str, default="my_korean_ppo_finetuned_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--input_max_text_length", type=int, default=12)
    parser.add_argument("--max_ppo_steps", type=int, default=20)
    
    args = parser.parse_args()
    print("Argparse 테스트 성공!")
    print(f"인자들: {args}")

if __name__ == "__main__":
    test_argparse()