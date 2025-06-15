"""
파인튜닝된 한국어 언어 모델 콘솔 채팅 테스트 스크립트
"""
import os
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_chat')

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='파인튜닝된 한국어 언어 모델 콘솔 채팅 테스트')
    
    parser.add_argument('--model-dir', 
                        help='파인튜닝된 모델 디렉토리 (예: my_optimized_model, my_korean_finetuned_model)')
    parser.add_argument('--base-model', default='EleutherAI/polyglot-ko-1.3b',
                        help='베이스 모델 이름')
    parser.add_argument('--max-length', type=int, default=512,
                        help='최대 시퀀스 길이')
    parser.add_argument('--max-new-tokens', type=int, default=128,
                        help='생성할 최대 새 토큰 수')
    parser.add_argument('--use-4bit', action='store_true',
                        help='4bit 양자화 사용')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='생성 온도 (낮을수록 보수적, 높을수록 창의적)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p 샘플링 값')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k 샘플링 값')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                        help='반복 페널티')
    
    return parser.parse_args()

def get_available_models():
    """현재 디렉토리에서 사용 가능한 파인튜닝 모델들 찾기"""
    current_dir = os.getcwd()
    potential_models = []
    
    # 일반적인 모델 디렉토리 이름들
    common_names = [
        'my_optimized_model', 'my_korean_finetuned_model', 
        'finetuned_model', 'trained_model', 'output'
    ]
    
    for name in common_names:
        if os.path.exists(name) and os.path.isdir(name):
            # 모델 파일들이 있는지 확인
            if (os.path.exists(os.path.join(name, 'adapter_config.json')) or 
                os.path.exists(os.path.join(name, 'pytorch_model.bin')) or
                os.path.exists(os.path.join(name, 'adapter_model.safetensors'))):
                potential_models.append(name)
    
    # 현재 디렉토리의 모든 디렉토리 검사
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if (os.path.isdir(item_path) and 
            item not in potential_models and
            not item.startswith('.') and
            item != '__pycache__'):
            # 모델 파일들이 있는지 확인
            if (os.path.exists(os.path.join(item_path, 'adapter_config.json')) or 
                os.path.exists(os.path.join(item_path, 'pytorch_model.bin')) or
                os.path.exists(os.path.join(item_path, 'adapter_model.safetensors'))):
                potential_models.append(item)
    
    return potential_models

def interactive_setup():
    """대화형 설정"""
    print("🤖 파인튜닝된 한국어 언어 모델 채팅 테스트")
    print("=" * 50)
    
    # 사용 가능한 모델 찾기
    available_models = get_available_models()
    
    if not available_models:
        print("❌ 파인튜닝된 모델을 찾을 수 없습니다!")
        print("다음 중 하나의 디렉토리가 있는지 확인하세요:")
        print("- my_optimized_model")
        print("- my_korean_finetuned_model")
        print("- 또는 다른 파인튜닝 모델 디렉토리")
        return None
    
    print(f"📁 발견된 모델 디렉토리: {len(available_models)}개")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    
    # 모델 선택
    while True:
        try:
            choice = input(f"\\n모델을 선택하세요 (1-{len(available_models)}): ").strip()
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(available_models):
                selected_model = available_models[model_idx]
                break
            else:
                print("❌ 잘못된 선택입니다. 다시 시도하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요.")
    
    # 기타 설정
    print(f"\\n✅ 선택된 모델: {selected_model}")
    
    # 최대 시퀀스 길이
    max_length = input("최대 시퀀스 길이 (기본값: 512): ").strip()
    max_length = int(max_length) if max_length else 512
    
    # 생성할 최대 토큰 수
    max_new_tokens = input("생성할 최대 토큰 수 (기본값: 128): ").strip()
    max_new_tokens = int(max_new_tokens) if max_new_tokens else 128
    
    # 생성 온도
    temperature = input("생성 온도 - 낮을수록 보수적 (기본값: 0.7): ").strip()
    temperature = float(temperature) if temperature else 0.7
    
    return {
        'model_dir': selected_model,
        'base_model': 'EleutherAI/polyglot-ko-1.3b',
        'max_length': max_length,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': 0.9,
        'top_k': 50,
        'repetition_penalty': 1.1,
        'use_4bit': False
    }

def load_model(config):
    """모델 로드"""
    print(f"\\n🔄 모델 로드 중...")
    print(f"📁 모델 디렉토리: {config['model_dir']}")
    print(f"🧠 베이스 모델: {config['base_model']}")
    
    # 양자화 설정
    if config['use_4bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("⚡ 4bit 양자화 사용")
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        print("⚡ 8bit 양자화 사용")
    
    try:
        # 베이스 모델 로드
        logger.info(f"베이스 모델 로드: {config['base_model']}")
        model = AutoModelForCausalLM.from_pretrained(
            config['base_model'],
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 파인튜닝된 어댑터 로드
        logger.info(f"파인튜닝 어댑터 로드: {config['model_dir']}")
        model = PeftModel.from_pretrained(model, config['model_dir'])
        model.eval()  # 추론 모드로 설정
        
        print("✅ 모델 로드 완료!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        print(f"❌ 모델 로드 실패: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, config):
    """응답 생성"""
    try:
        # 입력 토큰화
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=config['max_length'],
            truncation=True
        )
        
        # token_type_ids 제거 (있다면)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # GPU로 이동
        inputs = inputs.to(model.device)
        
        # 생성 설정
        generation_config = {
            'max_new_tokens': config['max_new_tokens'],
            'temperature': config['temperature'],
            'top_p': config['top_p'],
            'top_k': config['top_k'],
            'repetition_penalty': config['repetition_penalty'],
            'do_sample': True,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        # 텍스트 생성
        with torch.no_grad():
            start_time = time.time()
            generate_ids = model.generate(**inputs, **generation_config)
            generation_time = time.time() - start_time
        
        # 결과 디코딩
        response = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 원본 프롬프트 제거하여 생성된 부분만 반환
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # 생성 정보
        new_tokens = len(generate_ids[0]) - len(inputs['input_ids'][0])
        tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
        
        return response, {
            'generation_time': generation_time,
            'new_tokens': new_tokens,
            'tokens_per_sec': tokens_per_sec
        }
        
    except Exception as e:
        logger.error(f"응답 생성 실패: {e}")
        return f"❌ 생성 오류: {e}", None

def chat_loop(model, tokenizer, config):
    """채팅 루프"""
    print("\\n🎉 채팅을 시작합니다!")
    print("💡 팁:")
    print("  - 'quit', 'exit', '종료'를 입력하면 종료됩니다")
    print("  - 'clear'를 입력하면 화면을 지웁니다")
    print("  - 'config'를 입력하면 현재 설정을 확인할 수 있습니다")
    print("=" * 50)
    
    conversation_count = 0
    
    while True:
        try:
            # 사용자 입력
            user_input = input(f"\\n[{conversation_count + 1}] 사용자: ").strip()
            
            # 종료 명령
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\\n👋 채팅을 종료합니다.")
                break
            
            # 화면 지우기
            if user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            # 설정 확인
            if user_input.lower() == 'config':
                print("\\n⚙️ 현재 설정:")
                print(f"  📁 모델: {config['model_dir']}")
                print(f"  📏 최대 길이: {config['max_length']}")
                print(f"  🆕 최대 새 토큰: {config['max_new_tokens']}")
                print(f"  🌡️ 온도: {config['temperature']}")
                print(f"  📊 Top-p: {config['top_p']}")
                print(f"  🔝 Top-k: {config['top_k']}")
                print(f"  🔄 반복 페널티: {config['repetition_penalty']}")
                continue
            
            # 빈 입력 처리
            if not user_input:
                print("❌ 메시지를 입력하세요.")
                continue
            
            # 응답 생성
            print("🤖 AI: ", end="", flush=True)
            response, stats = generate_response(model, tokenizer, user_input, config)
            
            if response:
                print(response)
                
                # 생성 통계 출력
                if stats:
                    print(f"\\n📊 생성 정보: {stats['new_tokens']}토큰, "
                          f"{stats['generation_time']:.2f}초, "
                          f"{stats['tokens_per_sec']:.1f}토큰/초")
                
                conversation_count += 1
            
        except KeyboardInterrupt:
            print("\\n\\n👋 Ctrl+C로 종료됩니다.")
            break
        except Exception as e:
            logger.error(f"채팅 루프 오류: {e}")
            print(f"❌ 오류가 발생했습니다: {e}")

def main():
    """메인 함수"""
    # GPU 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다. GPU가 필요합니다.")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 명령줄 인자 확인
    args = parse_args()
    
    if args.model_dir:
        # 명령줄에서 모델 지정
        if not os.path.exists(args.model_dir):
            print(f"❌ 모델 디렉토리를 찾을 수 없습니다: {args.model_dir}")
            return
        
        config = vars(args)
    else:
        # 대화형 설정
        config = interactive_setup()
        if not config:
            return
    
    # 모델 로드
    model, tokenizer = load_model(config)
    if model is None or tokenizer is None:
        return
    
    # 채팅 시작
    try:
        chat_loop(model, tokenizer, config)
    except Exception as e:
        logger.error(f"메인 오류: {e}")
        print(f"❌ 오류가 발생했습니다: {e}")
    finally:
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\\n🧹 메모리 정리 완료")

if __name__ == "__main__":
    main()