"""
polyglot-ko 모델의 구조를 분석하여 적절한 LoRA 타겟 모듈을 찾는 스크립트
"""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn as nn

def analyze_model_structure():
    # 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    # 모델 로드
    print("모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/polyglot-ko-1.3b",
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("\n=== 모델 구조 분석 ===")
    print(f"모델 타입: {type(model)}")
    print(f"모델 설정: {model.config}")
    
    print("\n=== Linear 레이어 모듈들 ===")
    linear_modules = []
    attention_modules = []
    mlp_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append(name)
            print(f"Linear: {name}")
        elif 'attn' in name.lower():
            attention_modules.append(name)
            print(f"Attention: {name}")
        elif 'mlp' in name.lower():
            mlp_modules.append(name)
            print(f"MLP: {name}")
    
    print(f"\n총 Linear 모듈 수: {len(linear_modules)}")
    print(f"총 Attention 모듈 수: {len(attention_modules)}")
    print(f"총 MLP 모듈 수: {len(mlp_modules)}")
    
    # 추천 타겟 모듈
    recommended_targets = []
    
    # Linear 모듈의 마지막 부분 이름만 추출
    unique_names = set()
    for name in linear_modules[:20]:  # 처음 20개만 확인
        module_name = name.split('.')[-1]
        unique_names.add(module_name)
    
    print(f"\n추천 LoRA 타겟 모듈: {list(unique_names)}")
    
    return list(unique_names)

if __name__ == "__main__":
    analyze_model_structure()