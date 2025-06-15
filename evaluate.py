import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from datasets import load_dataset

class KoreanLLMEvaluator:
    def __init__(self, base_model_name="EleutherAI/polyglot-ko-1.3b", 
                 peft_model_path="my_optimized_model"):  # my_korean_finetuned_model
        self.base_model_name = base_model_name
        self.peft_model_path = peft_model_path
        self.load_models()
        self.setup_sentiment_analyzer()
        
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
        
    def setup_sentiment_analyzer(self):
        """감정 분석기를 설정합니다."""
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="monologg/koelectra-base-v3-discriminator",
            device_map="auto"
        )
        
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
    
    def analyze_sentiment(self, texts):
        """감정 분석을 수행합니다."""
        results = []
        for text in texts:
            try:
                sentiment = self.sentiment_pipe(text)
                # sentiment가 리스트인 경우 첫 번째 요소 사용
                if isinstance(sentiment, list):
                    sentiment = sentiment[0]
                
                # 라벨 매핑 (KoELECTRA 모델의 경우)
                label = sentiment.get('label', 'UNKNOWN')
                score = sentiment.get('score', 0.5)
                
                # 라벨이 POSITIVE/NEGATIVE인 경우와 LABEL_0/LABEL_1인 경우 모두 처리
                if label in ['POSITIVE', 'LABEL_1']:
                    is_positive = True
                elif label in ['NEGATIVE', 'LABEL_0']:
                    is_positive = False
                else:
                    # 스코어로 판단 (0.5 이상이면 긍정)
                    is_positive = score >= 0.5
                
                results.append({
                    'text': text,
                    'label': label,
                    'score': float(score),
                    'is_positive': is_positive
                })
                
            except Exception as e:
                print(f"감정 분석 오류 (텍스트: {text[:50]}...): {e}")
                results.append({
                    'text': text,
                    'label': 'ERROR',
                    'score': 0.5,  # 중립값으로 설정
                    'is_positive': False,
                    'error': str(e)
                })
        return results
    
    def compare_models(self, prompts, max_new_tokens=64, num_samples=3):
        """베이스 모델과 파인튜닝된 모델을 비교합니다."""
        results = {}
        
        for prompt in prompts:
            print(f"\n📝 프롬프트: '{prompt}'")
            print("-" * 60)
            
            # 베이스 모델 생성
            print("🔸 베이스 모델 결과:")
            base_texts = self.generate_text(prompt, self.base_model, max_new_tokens, num_samples)
            base_sentiments = self.analyze_sentiment(base_texts)
            
            for i, (text, sentiment) in enumerate(zip(base_texts, base_sentiments)):
                emotion = "😊 긍정" if sentiment['is_positive'] else "😞 부정"
                print(f"  {i+1}. {text}")
                print(f"     감정: {emotion} (신뢰도: {sentiment['score']:.3f})")
            
            # 파인튜닝된 모델 생성
            print("\n🔹 파인튜닝된 모델 결과:")
            ft_texts = self.generate_text(prompt, self.finetuned_model, max_new_tokens, num_samples)
            ft_sentiments = self.analyze_sentiment(ft_texts)
            
            for i, (text, sentiment) in enumerate(zip(ft_texts, ft_sentiments)):
                emotion = "😊 긍정" if sentiment['is_positive'] else "😞 부정"
                print(f"  {i+1}. {text}")
                print(f"     감정: {emotion} (신뢰도: {sentiment['score']:.3f})")
            
            # 통계 계산
            base_positive_ratio = sum(1 for s in base_sentiments if s['is_positive']) / len(base_sentiments)
            ft_positive_ratio = sum(1 for s in ft_sentiments if s['is_positive']) / len(ft_sentiments)
            
            # 유효한 스코어만 계산
            base_valid_scores = [s['score'] for s in base_sentiments if s['label'] != 'ERROR' and s['score'] > 0]
            ft_valid_scores = [s['score'] for s in ft_sentiments if s['label'] != 'ERROR' and s['score'] > 0]
            
            base_avg_score = sum(base_valid_scores) / len(base_valid_scores) if base_valid_scores else 0.5
            ft_avg_score = sum(ft_valid_scores) / len(ft_valid_scores) if ft_valid_scores else 0.5
            
            print(f"\n📊 통계:")
            print(f"  베이스 모델 - 긍정 비율: {base_positive_ratio:.1%}, 평균 신뢰도: {base_avg_score:.3f}")
            print(f"  파인튜닝 모델 - 긍정 비율: {ft_positive_ratio:.1%}, 평균 신뢰도: {ft_avg_score:.3f}")
            print(f"  개선도: {ft_positive_ratio - base_positive_ratio:+.1%}")
            
            results[prompt] = {
                'base_model': {
                    'texts': base_texts,
                    'sentiments': base_sentiments,
                    'positive_ratio': base_positive_ratio,
                    'avg_score': base_avg_score
                },
                'finetuned_model': {
                    'texts': ft_texts,
                    'sentiments': ft_sentiments,
                    'positive_ratio': ft_positive_ratio,
                    'avg_score': ft_avg_score
                },
                'improvement': ft_positive_ratio - base_positive_ratio
            }
            
        return results
    
    def evaluate_on_test_data(self, num_samples=10):
        """테스트 데이터에서 평가를 수행합니다."""
        print("\n🧪 테스트 데이터 평가 중...")
        
        try:
            # NSMC 테스트 데이터 로드
            test_dataset = load_dataset("nsmc", split="test", trust_remote_code=True)
            test_samples = test_dataset.select(range(num_samples))
            
            base_positive_count = 0
            ft_positive_count = 0
            
            for i, sample in enumerate(test_samples):
                prompt = sample['document'][:20] + "..."  # 앞 20자만 사용
                
                # 각 모델로 생성
                base_text = self.generate_text(prompt, self.base_model, max_new_tokens=32, num_samples=1)[0]
                ft_text = self.generate_text(prompt, self.finetuned_model, max_new_tokens=32, num_samples=1)[0]
                
                # 감정 분석
                base_sentiment = self.analyze_sentiment([base_text])[0]
                ft_sentiment = self.analyze_sentiment([ft_text])[0]
                
                if base_sentiment['is_positive']:
                    base_positive_count += 1
                if ft_sentiment['is_positive']:
                    ft_positive_count += 1
                
                print(f"샘플 {i+1}/{num_samples} 완료")
            
            base_positive_ratio = base_positive_count / num_samples
            ft_positive_ratio = ft_positive_count / num_samples
            
            print(f"\n📈 테스트 데이터 결과:")
            print(f"  베이스 모델 긍정 비율: {base_positive_ratio:.1%}")
            print(f"  파인튜닝 모델 긍정 비율: {ft_positive_ratio:.1%}")
            print(f"  개선도: {ft_positive_ratio - base_positive_ratio:+.1%}")
            
        except Exception as e:
            print(f"테스트 데이터 평가 중 오류: {e}")
    
    def save_results(self, results, filename="evaluation_results.json"):
        """결과를 JSON 파일로 저장합니다."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 결과가 {filename}에 저장되었습니다.")

def main():
    print("🚀 한국어 LLM 파인튜닝 평가 시작!")
    print("=" * 60)
    
    # 평가자 초기화
    evaluator = KoreanLLMEvaluator()
    
    # 테스트 프롬프트들
    test_prompts = [
        "이 영화는 정말",
        "배우들의 연기가",
        "스토리는",
        "영화관에서 보니",
        "감독의 연출이"
    ]
    
    # 모델 비교 평가
    print("\n🔍 모델 비교 평가")
    results = evaluator.compare_models(test_prompts, max_new_tokens=50, num_samples=3)
    
    # 테스트 데이터 평가
    evaluator.evaluate_on_test_data(num_samples=5)
    
    # 결과 저장
    evaluator.save_results(results)
    
    print("\n✅ 평가 완료!")
    
    # 전체 개선도 계산
    improvements = [result['improvement'] for result in results.values()]
    avg_improvement = sum(improvements) / len(improvements)
    print(f"\n🎯 전체 평균 개선도: {avg_improvement:+.1%}")

if __name__ == "__main__":
    main()