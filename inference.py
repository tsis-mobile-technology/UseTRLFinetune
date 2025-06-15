#!/usr/bin/env python3
"""
간단한 추론 스크립트 - 파인튜닝된 한국어 모델로 텍스트 생성
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse

class KoreanLLMInference:
    def __init__(self, base_model_name="EleutherAI/polyglot-ko-1.3b", 
                 peft_model_path="my_korean_finetuned_model"):
        self.base_model_name = base_model_name
        self.peft_model_path = self.check_model_path(peft_model_path)
        self.load_model()
        
    def check_model_path(self, model_path):
        """모델 경로를 확인하고 올바른 경로를 반환합니다."""
        import os
        
        # 주어진 경로가 존재하면 그대로 사용
        if os.path.exists(model_path):
            print(f"📁 모델 경로 확인: {model_path}")
            return model_path
        
        # 주요 경로들을 확인
        possible_paths = [
            "my_korean_finetuned_model",
            "my_optimized_model",
            "./my_korean_finetuned_model",
            "./my_optimized_model"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📁 대체 모델 경로 발견: {path}")
                return path
        
        # 모델이 없으면 에러 메시지
        print(f"⚠️ 경고: 모델 경로를 찾을 수 없습니다: {model_path}")
        print("📁 사용 가능한 모델:")
        for path in possible_paths:
            status = "✅ 존재" if os.path.exists(path) else "❌ 없음"
            print(f"  - {path}: {status}")
        
        # 기본값으로 계속 시도
        return model_path
        
    def load_model(self):
        """모델을 로드합니다."""
        print("🔄 모델 로딩 중...")
        
        try:
            # 양자화 설정
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # 베이스 모델 로드 (8비트 양자화 모델은 device_map="auto"로 설정)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"  # 8비트 양자화 모델을 위해 device_map 자동 설정
            )
            
            # 8비트 양자화 모델은 .cuda()를 호출하지 않음
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 파인튜닝된 어댑터 로드
            self.model = PeftModel.from_pretrained(self.model, self.peft_model_path)
            self.model.eval()
            
            print("✅ 모델 로딩 완료!")
            
        except Exception as e:
            print(f"❌ 모델 로딩 중 오류: {e}")
            print("🔄 대체 방법으로 시도 중...")
            
            # 대체 방법: device_map 없이 로드
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = PeftModel.from_pretrained(self.model, self.peft_model_path)
                self.model.eval()
                
                print("✅ 대체 방법으로 모델 로딩 완료!")
                
            except Exception as e2:
                print(f"❌ 대체 방법도 실패: {e2}")
                raise e2
        
    def generate(self, prompt, max_new_tokens=100, temperature=0.8, 
                top_k=50, top_p=0.95, repetition_penalty=2.0):
        """텍스트를 생성합니다."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # 입력을 모델과 동일한 장치로 이동
        # 모델이 양자화되어 있는지 여부와 관계없이 항상 모델과 동일한 디바이스에 있어야 함
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return result
    
    def interactive_mode(self):
        """대화형 모드를 실행합니다."""
        print("\n🤖 한국어 LLM 대화형 모드")
        print("=" * 50)
        print("프롬프트를 입력하세요 (종료: 'quit' 또는 'exit')")
        print("=" * 50)
        
        while True:
            try:
                prompt = input("\n💬 프롬프트: ").strip()
                
                if prompt.lower() in ['quit', 'exit', '종료', '나가기']:
                    print("👋 종료합니다!")
                    break
                
                if not prompt:
                    continue
                
                print("\n🔄 생성 중...")
                result = self.generate(prompt)
                print(f"\n🤖 생성 결과:\n{result}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 종료합니다!")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description="한국어 LLM 추론 스크립트")
    parser.add_argument("--prompt", type=str, help="생성할 프롬프트")
    parser.add_argument("--max_tokens", type=int, default=100, help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.8, help="생성 온도")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드 실행")
    parser.add_argument("--model_path", type=str, default="my_korean_finetuned_model", 
                       help="파인튜닝된 모델 경로")
    
    args = parser.parse_args()
    
    # 추론기 초기화
    llm = KoreanLLMInference(peft_model_path=args.model_path)
    
    if args.interactive:
        # 대화형 모드
        llm.interactive_mode()
    elif args.prompt:
        # 단일 프롬프트 모드
        print(f"📝 프롬프트: {args.prompt}")
        print("🔄 생성 중...")
        result = llm.generate(
            args.prompt, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"\n🤖 생성 결과:\n{result}")
    else:
        # 기본 테스트 프롬프트들
        test_prompts = [
            "이 영화는 정말",
            "배우들의 연기가",
            "스토리는 매우",
            "감독의 연출이",
            "전체적으로 이 작품은"
        ]
        
        print("🧪 테스트 프롬프트로 생성 중...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. 프롬프트: '{prompt}'")
            result = llm.generate(prompt, max_new_tokens=60)
            print(f"   결과: {result}")

if __name__ == "__main__":
    main()