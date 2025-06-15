#!/usr/bin/env python3
"""
빠른 테스트 스크립트 - 파인튜닝 효과 확인
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def quick_test():
    print("🔬 빠른 파인튜닝 효과 테스트")
    print("=" * 50)
    
    # 양자화 설정
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # 베이스 모델 로드
    print("🔄 베이스 모델 로딩...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/polyglot-ko-1.3b",
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 파인튜닝된 모델 로드
    print("🔄 파인튜닝된 모델 로딩...")
    finetuned_model = PeftModel.from_pretrained(base_model, "my_korean_finetuned_model")
    finetuned_model.eval()
    
    print("✅ 모델 로딩 완료!")
    
    # 테스트 프롬프트들
    test_prompts = [
        "이 영화는 정말",
        "배우들의 연기가",
        "스토리는",
        "감독의 연출이"
    ]
    
    generation_kwargs = {
        "max_new_tokens": 50,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 2.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # 긍정적 키워드 리스트
    positive_words = ['좋', '훌륭', '멋지', '재미있', '감동', '완벽', '최고', '놀라운', '뛰어난', '추천']
    
    for prompt in test_prompts:
        print(f"\n📝 프롬프트: '{prompt}'")
        print("-" * 40)
        
        # 베이스 모델 생성
        inputs = tokenizer(prompt, return_tensors="pt")
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        inputs = inputs.to("cuda")
        
        with torch.no_grad():
            # 베이스 모델
            base_output = base_model.generate(**inputs, **generation_kwargs)
            base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
            
            # 파인튜닝된 모델
            ft_output = finetuned_model.generate(**inputs, **generation_kwargs)
            ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)
        
        # 긍정 키워드 카운트
        base_positive_count = sum(1 for word in positive_words if word in base_text)
        ft_positive_count = sum(1 for word in positive_words if word in ft_text)
        
        print(f"🔸 베이스 모델: {base_text}")
        print(f"   긍정 키워드: {base_positive_count}개")
        
        print(f"🔹 파인튜닝 모델: {ft_text}")
        print(f"   긍정 키워드: {ft_positive_count}개")
        
        if ft_positive_count > base_positive_count:
            print("   ✅ 파인튜닝 효과: 더 긍정적!")
        elif ft_positive_count == base_positive_count:
            print("   ➖ 파인튜닝 효과: 비슷함")
        else:
            print("   ❌ 파인튜닝 효과: 덜 긍정적")
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    quick_test()