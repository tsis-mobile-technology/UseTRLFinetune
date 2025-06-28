#!/usr/bin/env python3
from trl import PPOTrainer
import inspect

# PPOTrainer의 __init__ 메서드 시그니처 확인
sig = inspect.signature(PPOTrainer.__init__)
print("PPOTrainer.__init__ 파라미터들:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
        if param.default != inspect.Parameter.empty:
            print(f"    기본값: {param.default}")

print("\nPPOTrainer 클래스의 docstring:")
print(PPOTrainer.__doc__)