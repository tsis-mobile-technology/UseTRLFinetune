#!/usr/bin/env python3

# 더 간단한 방법으로 PPOTrainer의 train 메소드 확인
import inspect
from trl import PPOTrainer

print("PPOTrainer의 train 메소드 시그니처:")
try:
    if hasattr(PPOTrainer, 'train'):
        sig = inspect.signature(PPOTrainer.train)
        print(f"train{sig}")
    else:
        print("train 메소드 없음")
except Exception as e:
    print(f"train 메소드 시그니처 확인 실패: {e}")

print("\nPPOTrainer의 step 메소드 시그니처:")
try:
    if hasattr(PPOTrainer, 'step'):
        sig = inspect.signature(PPOTrainer.step)
        print(f"step{sig}")
    else:
        print("step 메소드 없음")
except Exception as e:
    print(f"step 메소드 시그니처 확인 실패: {e}")

# 문서 문자열도 확인
print("\nPPOTrainer 문서:")
try:
    print(PPOTrainer.__doc__ if PPOTrainer.__doc__ else "문서 없음")
except:
    print("문서 확인 실패")

# TRL 버전
try:
    import trl
    print(f"\nTRL 버전: {trl.__version__}")
except:
    print("TRL 버전 확인 실패")