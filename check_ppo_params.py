"""
PPOTrainer의 매개변수 확인을 위한 테스트 스크립트
"""
import sys
from trl import PPOTrainer

# PPOTrainer 생성자의 매개변수 목록 출력
print("PPOTrainer 생성자 매개변수 목록:")
print(PPOTrainer.__init__.__code__.co_varnames)
print("\nPPOTrainer 생성자 매개변수 기본값:")
try:
    defaults = PPOTrainer.__init__.__defaults__
    args = PPOTrainer.__init__.__code__.co_varnames[1:len(defaults)+1]
    for arg, default in zip(args, defaults):
        print(f"{arg} = {default}")
except:
    print("기본값 확인 실패")

print("\nPPOTrainer 도움말:")
help(PPOTrainer)