#!/usr/bin/env python3
"""
간단한 평가 스크립트 - 키워드 기반 감정 분석
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re

class SimpleKoreanLLMEvaluator:
    def __init__(self, base_model_name="EleutherAI/polyglot-ko-1.3b", 
                 peft_model_path="my_korean_finetuned_model"):
        self.base_model_name = base_model_name
        self.peft_model_path = peft_model_path
        self.setup_sentiment_keywords()
        self.load_models()
        
    def setup_sentiment_keywords(self):
        """감정 분석을 위한 키워드를 설정합니다."""
        self.positive_keywords = [
            '좋', '훌륭', '멋지', '재미있', '흥미진진', '감동', '완벽', '최고',
            '놀라운', '뛰어난', '인상적', '아름다운', '환상적', '대단한', '우수한',
            '즐거운', '행복한', '만족', '추천', '걸작', '명작', '사랑', '좋아',
            '성공', '굉장', '끝내주', '대박', '쩐다', '개꿀', '갓', '레전드'
        ]
        
        self.negative_keywords = [
            '나쁘', '별로', '지루한', '실망', '최악', '엉망', '형편없', '시시한',
            '못하', '안좋', '구리', '루즈', '촌스러운', '유치한', '억지스러운',
            '어색한', '뻔한', '진부한', '식상한', '짜증', '답답한', '재미없', '싫어'
        ]
        
    def load_models(self):
        """베이스 모델과 파인튜닝된 모델을 로드합니다."""
        print("🔄 모델 로딩 중...")
        
        # 양자화 설정
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 베이스 모델 로드
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 파인튜닝된 모델 로드
        self.finetuned_model = PeftModel.from_pretrained(self.base_model, self.peft_model_path)
        self.finetuned_model.eval()
        
        print("✅ 모델 로딩 완료!")
        
    def analyze_sentiment_simple(self, text):
        """간단한 키워드 기반 감정 분석을 수행합니다."""
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        if positive_count > negative_count:
            return {'label': 'POSITIVE', 'score': 0.7 + (positive_count * 0.1), 'is_positive': True}\n        elif negative_count > positive_count:
            return {'label': 'NEGATIVE', 'score': 0.7 + (negative_count * 0.1), 'is_positive': False}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5, 'is_positive': False}
    
    def generate_text(self, prompt, model, max_new_tokens=64, num_samples=1):
        """텍스트를 생성합니다."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        inputs = inputs.to("cuda")
        
        generated_texts = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=2.0,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            generated_texts.append(result)
            
        return generated_texts
    
    def compare_models_simple(self, prompts, max_new_tokens=64, num_samples=3):
        """베이스 모델과 파인튜닝된 모델을 간단히 비교합니다."""
        results = {}
        
        for prompt in prompts:
            print(f"\n📝 프롬프트: '{prompt}'")
            print("-" * 60)
            
            # 베이스 모델 생성
            print("🔸 베이스 모델 결과:")
            base_texts = self.generate_text(prompt, self.base_model, max_new_tokens, num_samples)
            
            base_sentiments = []
            for i, text in enumerate(base_texts):
                sentiment = self.analyze_sentiment_simple(text)
                base_sentiments.append(sentiment)
                emotion = "😊 긍정" if sentiment['is_positive'] else "😞 부정/중립"
                print(f"  {i+1}. {text}")
                print(f"     감정: {emotion} (신뢰도: {sentiment['score']:.3f})")
            
            # 파인튜닝된 모델 생성
            print("\n🔹 파인튜닝된 모델 결과:")
            ft_texts = self.generate_text(prompt, self.finetuned_model, max_new_tokens, num_samples)
            
            ft_sentiments = []
            for i, text in enumerate(ft_texts):
                sentiment = self.analyze_sentiment_simple(text)
                ft_sentiments.append(sentiment)
                emotion = "😊 긍정" if sentiment['is_positive'] else "😞 부정/중립"
                print(f"  {i+1}. {text}")
                print(f"     감정: {emotion} (신뢰도: {sentiment['score']:.3f})")
            
            # 통계 계산
            base_positive_ratio = sum(1 for s in base_sentiments if s['is_positive']) / len(base_sentiments)
            ft_positive_ratio = sum(1 for s in ft_sentiments if s['is_positive']) / len(ft_sentiments)
            
            base_avg_score = sum(s['score'] for s in base_sentiments) / len(base_sentiments)
            ft_avg_score = sum(s['score'] for s in ft_sentiments) / len(ft_sentiments)
            
            print(f"\n📊 통계:")
            print(f"  베이스 모델 - 긍정 비율: {base_positive_ratio:.1%}, 평균 신뢰도: {base_avg_score:.3f}")
            print(f"  파인튜닝 모델 - 긍정 비율: {ft_positive_ratio:.1%}, 평균 신뢰도: {ft_avg_score:.3f}")
            print(f"  개선도: {ft_positive_ratio - base_positive_ratio:+.1%}")
            
            results[prompt] = {
                'base_positive_ratio': base_positive_ratio,
                'ft_positive_ratio': ft_positive_ratio,
                'improvement': ft_positive_ratio - base_positive_ratio,
                'base_avg_score': base_avg_score,
                'ft_avg_score': ft_avg_score
            }
            
        return results

def main():
    print("🚀 한국어 LLM 간단 평가 시작!")
    print("=" * 60)
    
    # 평가자 초기화
    evaluator = SimpleKoreanLLMEvaluator()
    
    # 테스트 프롬프트들
    test_prompts = [
        "이 영화는 정말",
        "배우들의 연기가",
        "스토리는",
        "영화관에서 보니",
        "감독의 연출이"
    ]
    
    # 모델 비교 평가
    print("\n🔍 모델 비교 평가 (키워드 기반)")
    results = evaluator.compare_models_simple(test_prompts, max_new_tokens=50, num_samples=3)
    
    print("\n✅ 평가 완료!")
    
    # 전체 개선도 계산
    improvements = [result['improvement'] for result in results.values()]
    avg_improvement = sum(improvements) / len(improvements)
    print(f"\n🎯 전체 평균 개선도: {avg_improvement:+.1%}")
    
    # 긍정 비율 요약
    print(f"\n📈 긍정 비율 요약:")
    for prompt, result in results.items():
        print(f"  '{prompt}': {result['base_positive_ratio']:.0%} → {result['ft_positive_ratio']:.0%} ({result['improvement']:+.0%})")

if __name__ == "__main__":
    main()