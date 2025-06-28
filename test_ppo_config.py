#!/usr/bin/env python3
from trl import PPOConfig
import inspect

# PPOConfig의 __init__ 메서드 시그니처 확인
sig = inspect.signature(PPOConfig.__init__)
print("PPOConfig.__init__ 파라미터들:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
        if param.default != inspect.Parameter.empty:
            print(f"    기본값: {param.default}")

print("\n기본 PPOConfig 생성 테스트:")
try:
    config = PPOConfig()
    print("✅ 기본 PPOConfig 생성 성공")
    print(f"기본 설정들: {config}")
except Exception as e:
    print(f"❌ 기본 PPOConfig 생성 실패: {e}")

print("\n최소 파라미터로 PPOConfig 생성 테스트:")
try:
    config = PPOConfig(learning_rate=1e-5, batch_size=4, mini_batch_size=2)
    print("✅ 최소 파라미터 PPOConfig 생성 성공")
except Exception as e:
    print(f"❌ 최소 파라미터 PPOConfig 생성 실패: {e}")