#!/usr/bin/env python3

# TRL 버전 및 PPOTrainer 메소드 확인
try:
    import trl
    print(f"TRL 버전: {trl.__version__}")
except:
    print("TRL 버전 확인 실패")

try:
    from trl import PPOTrainer
    print(f"PPOTrainer 클래스: {PPOTrainer}")
    
    # 클래스 메소드 확인 (인스턴스 생성 없이)
    methods = [method for method in dir(PPOTrainer) if not method.startswith('_')]
    print("\nPPOTrainer 클래스 메소드들:")
    for method in sorted(methods):
        print(f"  - {method}")
        
    # 특히 중요한 메소드들 확인
    important_methods = ['step', 'train', 'update', 'train_step', 'ppo_step', 'compute_loss']
    print("\n중요한 메소드들 존재 여부:")
    for method in important_methods:
        exists = hasattr(PPOTrainer, method)
        print(f"  - {method}: {'✅' if exists else '❌'}")
        
except Exception as e:
    print(f"PPOTrainer 임포트 실패: {e}")

# transformers 버전도 확인
try:
    import transformers
    print(f"\nTransformers 버전: {transformers.__version__}")
except:
    print("Transformers 버전 확인 실패")

# torch 버전도 확인  
try:
    import torch
    print(f"PyTorch 버전: {torch.__version__}")
except:
    print("PyTorch 버전 확인 실패")