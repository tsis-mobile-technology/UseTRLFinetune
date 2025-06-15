"""
데이터 수집 및 모델 훈련 데모 스크립트
"""
import os
import argparse
import logging
import subprocess
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo')

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='데이터 수집 및 모델 훈련 데모')
    
    parser.add_argument('--demo-mode', choices=['web', 'pdf', 'supabase', 'all'], default='all',
                        help='실행할 데모 모드 선택')
    parser.add_argument('--output-dir', default='demo_output',
                        help='데모 출력 디렉토리')
    parser.add_argument('--skip-training', action='store_true',
                        help='모델 훈련 단계 건너뛰기 (데이터 수집만 실행)')
    
    return parser.parse_args()

def setup_demo_environment(output_dir):
    """데모 환경 설정"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f"데모 환경 설정 완료: {output_dir}")
    return data_dir

def demo_web_scraping(data_dir):
    """웹 스크래핑 데모"""
    logger.info("웹 스크래핑 데모 시작")
    
    # 데모용 URL 목록
    urls = [
        "https://ko.wikipedia.org/wiki/인공지능",
        "https://ko.wikipedia.org/wiki/기계학습",
        "https://ko.wikipedia.org/wiki/자연어_처리"
    ]
    
    # URL을 문자열로 변환
    urls_str = ' '.join([f'"{url}"' for url in urls])
    
    # 명령 실행
    output_path = os.path.join(data_dir, 'web_data.jsonl')
    cmd = f'python main.py --collect --web {urls_str} --output "{output_path}"'
    
    logger.info(f"실행 명령: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"웹 스크래핑 완료: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"웹 스크래핑 실패: {e}")
        return None

def demo_pdf_extraction(data_dir):
    """PDF 추출 데모"""
    logger.info("PDF 추출 데모 시작")
    
    # 데모용 PDF URL 목록
    urls = [
        "https://arxiv.org/pdf/2005.14165.pdf",  # 영어 PDF 예시
        "https://arxiv.org/pdf/2201.08239.pdf"   # 다른 PDF 예시
    ]
    
    # URL을 문자열로 변환
    urls_str = ' '.join([f'"{url}"' for url in urls])
    
    # 명령 실행
    output_path = os.path.join(data_dir, 'pdf_data.jsonl')
    cmd = f'python main.py --collect --pdf {urls_str} --output "{output_path}"'
    
    logger.info(f"실행 명령: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"PDF 추출 완료: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"PDF 추출 실패: {e}")
        return None

def demo_supabase_integration(data_dir):
    """Supabase 통합 데모"""
    logger.info("Supabase 통합 데모 시작")
    
    # Supabase 자격 증명 확인
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        logger.warning("Supabase 자격 증명이 설정되지 않았습니다. 환경 변수 SUPABASE_URL 및 SUPABASE_KEY를 설정하세요.")
        logger.warning("Supabase 데모를 건너뜁니다.")
        return None
    
    # 텍스트 파일 생성 (데모용)
    text_file = os.path.join(data_dir, 'sample.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("""
        인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력, 자연언어의 이해능력 등을 인공적으로 구현하는 컴퓨터 공학 분야이다.
        
        데이터를 기반으로 학습하는 머신러닝과 딥러닝은 현대 인공지능 기술의 핵심이다.
        
        이 기술을 통해 컴퓨터는 인간의 개입 없이도 데이터로부터 패턴을 학습하고 의사결정을 내릴 수 있게 되었다.
        """)
    
    # 명령 실행
    output_path = os.path.join(data_dir, 'supabase_data.jsonl')
    cmd = (f'python main.py --collect --text "{text_file}" --supabase '
           f'--supabase-url "{supabase_url}" --supabase-key "{supabase_key}" '
           f'--export-supabase "{output_path}"')
    
    logger.info("Supabase 명령 실행 중...")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Supabase 통합 완료: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Supabase 통합 실패: {e}")
        return None

def train_model_with_data(data_paths, output_dir):
    """수집한 데이터로 모델 훈련"""
    if not data_paths or len(data_paths) == 0:
        logger.error("훈련할 데이터가 없습니다.")
        return False
    
    # 모든 데이터 파일 병합
    merged_data_path = os.path.join(output_dir, 'all_training_data.jsonl')
    
    with open(merged_data_path, 'w', encoding='utf-8') as out_file:
        for data_path in data_paths:
            if data_path and os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as in_file:
                    out_file.write(in_file.read())
    
    logger.info(f"훈련 데이터 병합 완료: {merged_data_path}")
    
    # 훈련 명령 실행
    model_output_dir = os.path.join(output_dir, 'finetuned_model')
    cmd = (f'python custom_train.py --data-path "{merged_data_path}" '
           f'--output-dir "{model_output_dir}" --epochs 1 --limit-samples 50')
    
    logger.info(f"모델 훈련 명령: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"모델 훈련 완료: {model_output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"모델 훈련 실패: {e}")
        return False

def main():
    """메인 함수"""
    args = parse_args()
    
    # 데모 환경 설정
    data_dir = setup_demo_environment(args.output_dir)
    
    # 데이터 수집 결과 저장
    data_paths = []
    
    # 선택한 데모 모드에 따라 실행
    if args.demo_mode in ['web', 'all']:
        web_data_path = demo_web_scraping(data_dir)
        if web_data_path:
            data_paths.append(web_data_path)
    
    if args.demo_mode in ['pdf', 'all']:
        pdf_data_path = demo_pdf_extraction(data_dir)
        if pdf_data_path:
            data_paths.append(pdf_data_path)
    
    if args.demo_mode in ['supabase', 'all']:
        supabase_data_path = demo_supabase_integration(data_dir)
        if supabase_data_path:
            data_paths.append(supabase_data_path)
    
    # 모델 훈련 (필요한 경우)
    if not args.skip_training and data_paths:
        train_success = train_model_with_data(data_paths, args.output_dir)
        if train_success:
            logger.info("데모가 성공적으로 완료되었습니다.")
            logger.info(f"훈련된 모델: {os.path.join(args.output_dir, 'finetuned_model')}")
        else:
            logger.error("모델 훈련 단계에서 데모가 실패했습니다.")
    elif args.skip_training:
        logger.info("모델 훈련을 건너뛰고 데이터 수집만 완료했습니다.")
    else:
        logger.warning("수집된 데이터가 없어 모델 훈련을 건너뜁니다.")

if __name__ == "__main__":
    main()