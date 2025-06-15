#!/usr/bin/env python3
"""
안정적인 추론 스크립트 - 파인튜닝된 한국어 모델로 텍스트 생성
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
import os

class SafeKoreanLLMInference:
    def __init__(self, base_model_name="EleutherAI/polyglot-ko-1.3b", 
                 peft_model_path="my_korean_finetuned_model"):
        self.base_model_name = base_model_name
        self.peft_model_path = self.find_model_path(peft_model_path)
        self.load_model()
        
    def find_model_path(self, model_path):
        """모델 경로를 찾습니다."""
        possible_paths = [
            model_path,
            "my_korean_finetuned_model",
            "my_optimized_model",
            "./my_korean_finetuned_model",
            "./my_optimized_model"
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                # adapter_config.json 파일이 있는지 확인
                if os.path.exists(os.path.join(path, "adapter_config.json")):
                    print(f"✅ 모델 발견: {path}")
                    return path
        
        print(f"❌ 모델을 찾을 수 없습니다. 다음 경로들을 확인했습니다:")
        for path in possible_paths:
            print(f"  - {path}: {'존재' if os.path.exists(path) else '없음'}")
        
        raise FileNotFoundError(f"모델 디렉터리를 찾을 수 없습니다.")
        
    def load_model(self):
        """모델을 로드합니다."""
        print("🔄 모델 로딩 중...")
        
        try:
            # 단계별로 안전하게 로드
            print("  1. 베이스 모델 로딩...")
            
            # 먼저 양자화 없이 시도
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            print("  2. 토크나이저 로딩...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("  3. PEFT 어댑터 로딩...")
            self.model = PeftModel.from_pretrained(self.model, self.peft_model_path)
            
            print("  4. GPU로 이동...")
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model.eval()
            print("✅ 모델 로딩 완료!")
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print("🔄 양자화 없이 재시도 중...")
            
            try:
                # 양자화 없이 CPU에서 로드
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = PeftModel.from_pretrained(self.model, self.peft_model_path)
                self.model.eval()
                
                print("✅ CPU 모드로 모델 로딩 완료!")\n                
            except Exception as e2:
                print(f"❌ 재시도도 실패: {e2}")
                raise e2
        
    def generate(self, prompt, max_new_tokens=100, temperature=0.8, 
                top_k=50, top_p=0.95, repetition_penalty=2.0):
        """텍스트를 생성합니다."""
        print(f"🔄 '{prompt}' 생성 중...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # 모델이 CUDA에 있으면 입력도 CUDA로
        if next(self.model.parameters()).is_cuda:
            inputs = inputs.to("cuda")
        
        try:
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
            
        except Exception as e:
            print(f"❌ 생성 중 오류: {e}")
            return f"생성 실패: {prompt}"
    
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
                
                result = self.generate(prompt)
                print(f"\n🤖 생성 결과:\n{result}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 종료합니다!")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def main():
    parser = argparse.ArgumentParser(description="안정적인 한국어 LLM 추론 스크립트")
    parser.add_argument("--prompt", type=str, help="생성할 프롬프트")
    parser.add_argument("--max_tokens", type=int, default=100, help="최대 생성 토큰 수")
    parser.add_argument("--temperature", type=float, default=0.8, help="생성 온도")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드 실행")
    parser.add_argument("--model_path", type=str, default="my_korean_finetuned_model", 
                       help="파인튜닝된 모델 경로")
    
    args = parser.parse_args()
    
    try:
        # 추론기 초기화
        llm = SafeKoreanLLMInference(peft_model_path=args.model_path)
        
        if args.interactive:
            # 대화형 모드
            llm.interactive_mode()
        elif args.prompt:
            # 단일 프롬프트 모드
            print(f"📝 프롬프트: {args.prompt}")
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
                "스토리는 매우"
            ]
            
            print("🧪 테스트 프롬프트로 생성 중...")
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n{i}. 프롬프트: '{prompt}'")
                result = llm.generate(prompt, max_new_tokens=60)
                print(f"   결과: {result}")
    
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        print("\n🔧 문제 해결 방법:")
        print("1. 모델 경로를 확인하세요")
        print("2. CUDA 메모리가 부족한 경우 다른 프로그램을 종료하세요")
        print("3. 가상환경이 활성화되어 있는지 확인하세요")

if __name__ == "__main__":
    main()