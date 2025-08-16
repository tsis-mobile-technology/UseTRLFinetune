#!/usr/bin/env python3
"""
병합된 모델을 GGUF 형식으로 변환하여 Ollama에서 사용 가능하도록 만드는 스크립트
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import json
import shutil

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gguf_conversion.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_llama_cpp_availability():
    """llama.cpp 도구 가용성 확인"""
    logger = logging.getLogger(__name__)
    
    # 일반적인 llama.cpp 위치들 확인
    possible_paths = [
        "llama.cpp/convert_hf_to_gguf.py",
        "./convert_hf_to_gguf.py",
        "convert-hf-to-gguf.py",
        "/usr/local/bin/convert-hf-to-gguf.py",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            logger.info(f"✅ llama.cpp 변환 스크립트 발견: {path}")
            return str(Path(path).resolve())
    
    # PATH에서 확인
    try:
        result = subprocess.run(["which", "convert-hf-to-gguf.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            path = result.stdout.strip()
            logger.info(f"✅ llama.cpp 변환 스크립트 발견 (PATH): {path}")
            return path
    except:
        pass
    
    logger.warning("❌ llama.cpp 변환 스크립트를 찾을 수 없습니다.")
    return None

def install_llama_cpp():
    """llama.cpp 자동 설치"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("📥 llama.cpp 클론 중...")
        
        # llama.cpp 저장소 클론
        if not Path("llama.cpp").exists():
            subprocess.run([
                "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
            ], check=True)
            logger.info("✅ llama.cpp 클론 완료")
        else:
            logger.info("✅ llama.cpp 이미 존재함")
        
        # 요구사항 설치
        logger.info("📦 Python 의존성 설치 중...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "transformers", "sentencepiece", "gguf"
        ], check=True)
        
        return "llama.cpp/convert_hf_to_gguf.py"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ llama.cpp 설치 실패: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        return None

def convert_to_gguf(
    model_path: str = "my_korean_gpt_oss_20b_merged",
    output_path: str = "my_korean_gpt_oss_20b.gguf",
    vocab_type: str = "bpe",
    quantization: str = "f16"
):
    """HuggingFace 모델을 GGUF로 변환"""
    logger = setup_logging()
    
    try:
        logger.info("🚀 GGUF 변환 프로세스 시작")
        
        # 입력 모델 경로 확인
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_path}")
        
        logger.info(f"📁 입력 모델: {model_path}")
        logger.info(f"📄 출력 파일: {output_path}")
        
        # llama.cpp 변환 스크립트 확인
        converter_script = check_llama_cpp_availability()
        
        if not converter_script:
            logger.info("🔧 llama.cpp 자동 설치 시도...")
            converter_script = install_llama_cpp()
            
            if not converter_script:
                raise RuntimeError("llama.cpp 변환 도구를 설치할 수 없습니다.")
        
        # GGUF 변환 명령어 구성
        cmd = [
            sys.executable,
            converter_script,
            str(model_path),
            "--outfile", output_path,
        ]
        
        # 양자화 옵션 추가 (지원되는 형식만)
        supported_types = ["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"]
        if quantization in supported_types:
            cmd.extend(["--outtype", quantization])
        else:
            logger.warning(f"⚠️ 지원되지 않는 양자화 타입 {quantization}, f16으로 변경")
            cmd.extend(["--outtype", "f16"])
        
        logger.info(f"🔄 변환 명령어: {' '.join(cmd)}")
        logger.info("⏳ 변환 중... (수 분 소요될 수 있습니다)")
        
        # 변환 실행
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ GGUF 변환 성공!")
            
            # 출력 파일 크기 확인
            if Path(output_path).exists():
                file_size = Path(output_path).stat().st_size / 1024**3
                logger.info(f"📊 GGUF 파일 크기: {file_size:.1f}GB")
                
                # 메타데이터 파일 생성
                create_gguf_metadata(output_path, model_path, quantization)
                
                return True
            else:
                logger.error("❌ 출력 파일이 생성되지 않았습니다.")
                return False
        else:
            logger.error(f"❌ 변환 실패: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
            return False
            
    except Exception as e:
        logger.error(f"❌ GGUF 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_gguf_metadata(gguf_path: str, model_path: Path, quantization: str):
    """GGUF 파일에 대한 메타데이터 생성"""
    logger = logging.getLogger(__name__)
    
    try:
        # 원본 모델 정보 로드
        model_info = {}
        info_file = model_path / "model_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
        
        # GGUF 메타데이터 생성
        gguf_metadata = {
            "format": "GGUF",
            "quantization": quantization,
            "source_model": str(model_path),
            "gguf_file": gguf_path,
            "created_at": "2025-08-16",
            "compatible_with": ["ollama", "llama.cpp", "oobabooga"],
            "korean_finetuned": True,
            "original_info": model_info
        }
        
        # 메타데이터 파일 저장
        metadata_path = Path(gguf_path).with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(gguf_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 메타데이터 저장: {metadata_path}")
        
    except Exception as e:
        logger.error(f"❌ 메타데이터 생성 실패: {e}")

def create_modelfile_for_ollama(
    gguf_path: str,
    model_name: str = "korean-gpt-oss-20b",
    output_path: str = "Modelfile"
):
    """Ollama용 Modelfile 생성"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("📝 Ollama Modelfile 생성 중...")
        
        # GGUF 파일의 절대 경로 구하기
        gguf_absolute_path = Path(gguf_path).resolve()
        
        # Modelfile 내용 생성
        modelfile_content = f"""# Korean Fine-tuned GPT-OSS-20B Model
FROM {gguf_absolute_path}

# 한국어 최적화 설정
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# 시스템 프롬프트 (한국어 대화)
SYSTEM \"\"\"당신은 한국어를 유창하게 구사하는 AI 어시스턴트입니다. 
사용자의 질문에 정확하고 도움이 되는 답변을 한국어로 제공해주세요.
예의 바르고 친근한 톤으로 대화해주세요.\"\"\"

# 채팅 템플릿 (기본 대화형)
TEMPLATE \"\"\"사용자: {{{{ .Prompt }}}}
어시스턴트: {{{{ .Response }}}}\"\"\"
"""

        # Modelfile 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"✅ Modelfile 생성 완료: {output_path}")
        logger.info(f"🚀 Ollama 모델 생성 명령어:")
        logger.info(f"   ollama create {model_name} -f {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Modelfile 생성 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="HuggingFace 모델을 GGUF로 변환하고 Ollama용 설정 생성")
    parser.add_argument("--model-path", type=str, default="my_korean_gpt_oss_20b_merged",
                       help="변환할 HuggingFace 모델 경로")
    parser.add_argument("--output", type=str, default="my_korean_gpt_oss_20b.gguf",
                       help="출력 GGUF 파일명")
    parser.add_argument("--quantization", type=str, default="f16",
                       choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"],
                       help="양자화 레벨")
    parser.add_argument("--model-name", type=str, default="korean-gpt-oss-20b",
                       help="Ollama 모델명")
    parser.add_argument("--skip-conversion", action="store_true",
                       help="GGUF 변환 건너뛰고 Modelfile만 생성")
    
    args = parser.parse_args()
    
    print("🔄 GGUF 변환 및 Ollama 통합 도구")
    print(f"📁 입력 모델: {args.model_path}")
    print(f"📄 출력 GGUF: {args.output}")
    print(f"⚙️ 양자화: {args.quantization}")
    print(f"🏷️ 모델명: {args.model_name}")
    print()
    
    success = True
    
    if not args.skip_conversion:
        # GGUF 변환 실행
        success = convert_to_gguf(
            model_path=args.model_path,
            output_path=args.output,
            quantization=args.quantization
        )
    
    if success:
        # Ollama Modelfile 생성
        modelfile_success = create_modelfile_for_ollama(
            gguf_path=args.output,
            model_name=args.model_name,
            output_path="Modelfile"
        )
        
        if modelfile_success:
            print("\n🎉 변환 및 설정 완료!")
            print(f"📄 GGUF 파일: {args.output}")
            print(f"📝 Modelfile: Modelfile")
            print("\n🚀 Ollama에서 사용하기:")
            print(f"   ollama create {args.model_name} -f Modelfile")
            print(f"   ollama run {args.model_name}")
        else:
            print("\n⚠️ GGUF 변환은 성공했지만 Modelfile 생성에 실패했습니다.")
    else:
        print("\n❌ GGUF 변환 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()