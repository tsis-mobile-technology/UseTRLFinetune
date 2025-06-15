"""
PPOTrainer의 매개변수 확인을 위한 간단한 테스트
"""
import inspect
from trl import PPOTrainer, PPOConfig

# PPOTrainer의 __init__ 메서드 시그니처 확인
sig = inspect.signature(PPOTrainer.__init__)
print("PPOTrainer.__init__ 매개변수:")
for param_name, param in sig.parameters.items():
    print(f"  {param_name}: {param}")

print("\n")

# PPOConfig 객체 생성 테스트
try:
    config = PPOConfig()
    print("PPOConfig 생성 성공")
    print(f"PPOConfig 속성들: {dir(config)}")
except Exception as e:
    print(f"PPOConfig 생성 실패: {e}")

# PPOTrainer 생성 테스트 (최소한의 인자로)
try:
    # 실제로는 모델이 필요하지만, 인자 확인을 위해서만
    print("\nPPOTrainer 생성 테스트...")
    # PPOTrainer(model=None)  # 이것만으로 어떤 인자가 필요한지 확인 가능
except Exception as e:
    print(f"PPOTrainer 생성 에러: {e}")
    print("이 에러 메시지에서 필요한 인자를 확인할 수 있습니다.")