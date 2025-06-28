"""
PDF 파일에서 텍스트를 추출하는 모듈
"""
import os
import requests
import logging
import io
import re
import tempfile
from urllib.parse import urlparse
import PyPDF2
import pdfplumber

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pdf_extractor')

def download_pdf(url):
    """
    URL에서 PDF 파일 다운로드
    
    Args:
        url (str): PDF 파일 URL
        
    Returns:
        bytes: PDF 파일 바이너리 데이터
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        
        # Content-Type 확인
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
            logger.warning(f"PDF가 아닐 수 있음: {url} (Content-Type: {content_type})")
        
        return response.content
    
    except Exception as e:
        logger.error(f"PDF 다운로드 중 오류 발생: {str(e)}")
        return None

def extract_text_with_pypdf2(pdf_content):
    """
    PyPDF2를 사용하여 PDF에서 텍스트 추출
    
    Args:
        pdf_content (bytes): PDF 파일 바이너리 데이터
        
    Returns:
        str: 추출된 텍스트
    """
    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # 각 페이지에서 텍스트 추출
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        return text
    
    except PyPDF2.errors.FileNotDecryptedError:
        logger.error(f"PyPDF2: 암호화된 PDF 파일이라 텍스트를 추출할 수 없습니다.")
        return None
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PyPDF2: PDF 읽기 오류 발생: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"PyPDF2: 텍스트 추출 중 예상치 못한 오류 발생: {str(e)}")
        return None

def extract_text_with_pdfplumber(pdf_content):
    """
    pdfplumber를 사용하여 PDF에서 텍스트 추출
    
    Args:
        pdf_content (bytes): PDF 파일 바이너리 데이터
        
    Returns:
        str: 추출된 텍스트
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=1, y_tolerance=3) # x_tolerance, y_tolerance 추가하여 추출 정확도 향상 시도
                if page_text:
                    text += page_text + "\n\n"
            return text
    
    except pdfplumber.exceptions.PasswordRequired:
        logger.error(f"pdfplumber: 암호화된 PDF 파일이라 텍스트를 추출할 수 없습니다.")
        return None
    except pdfplumber.exceptions.PDFSyntaxError as e:
        logger.error(f"pdfplumber: PDF 문법 오류 발생: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"pdfplumber: 텍스트 추출 중 예상치 못한 오류 발생: {str(e)}")
        return None

def clean_text(text):
    """
    추출된 텍스트 정리
    
    Args:
        text (str): 정리할 텍스트
        
    Returns:
        str: 정리된 텍스트
    """
    if not text:
        return ""
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 불필요한 특수문자 정리
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\'\"\(\)\[\]\{\}]', '', text)
    
    # 빈 줄 제거하고 한 줄로 합치기
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = ' '.join(lines)
    
    return text.strip()

def extract_text_from_pdf_url(url):
    """
    URL에서 PDF 다운로드하고 텍스트 추출
    
    Args:
        url (str): PDF 파일 URL
        
    Returns:
        dict: 추출된 텍스트와 메타데이터
    """
    logger.info(f"PDF URL에서 텍스트 추출 시작: {url}")
    
    # PDF 다운로드
    pdf_content = download_pdf(url)
    if not pdf_content:
        return None
    
    # 파일명 추출
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded.pdf"
    
    # 여러 방법으로 텍스트 추출 시도
    text_pypdf2 = extract_text_with_pypdf2(pdf_content)
    text_pdfplumber = extract_text_with_pdfplumber(pdf_content)
    
    # 더 많은 텍스트를 추출한 방법 선택
    if text_pdfplumber and (not text_pypdf2 or len(text_pdfplumber) > len(text_pypdf2)):
        text = text_pdfplumber
        method = 'pdfplumber'
    else:
        text = text_pypdf2
        method = 'pypdf2'
    
    if not text:
        logger.error(f"PDF에서 텍스트 추출 실패: {url}")
        return None
    
    # 텍스트 정리
    cleaned_text = clean_text(text)
    
    return {
        'text': cleaned_text,
        'url': url,
        'filename': filename,
        'method': method,
        'length': len(cleaned_text)
    }

def extract_text_from_pdf_file(file_path):
    """
    로컬 PDF 파일에서 텍스트 추출
    
    Args:
        file_path (str): PDF 파일 경로
        
    Returns:
        dict: 추출된 텍스트와 메타데이터
    """
    logger.info(f"로컬 PDF 파일에서 텍스트 추출 시작: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            pdf_content = f.read()

    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e: # Other potential IO errors
        logger.error(f"파일 읽기 중 오류 발생 ({file_path}): {str(e)}")
        return None
        
    # 파일 읽기 성공 후 텍스트 추출 시도
    try:
        # 여러 방법으로 텍스트 추출 시도
        text_pypdf2 = extract_text_with_pypdf2(pdf_content)
        text_pdfplumber = extract_text_with_pdfplumber(pdf_content)
        
        # 더 많은 텍스트를 추출한 방법 선택
        if text_pdfplumber and (not text_pypdf2 or len(text_pdfplumber) > len(text_pypdf2)):
            text = text_pdfplumber
            method = 'pdfplumber'
        else:
            text = text_pypdf2
            method = 'pypdf2'
        
        if not text:
            logger.error(f"PDF에서 텍스트 추출 실패: {file_path}")
            return None
        
        # 텍스트 정리
        cleaned_text = clean_text(text)
        
        return {
            'text': cleaned_text,
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'method': method,
            'length': len(cleaned_text)
        }
    
    except Exception as e: # Catch errors during extraction if any, though specific ones are in sub-functions
        logger.error(f"PDF 내용 처리 중 오류 발생 ({file_path}): {str(e)}")
        return None

def process_pdf_urls(urls):
    """
    여러 PDF URL에서 텍스트 추출
    
    Args:
        urls (list): 텍스트를 추출할 PDF URL 목록
        
    Returns:
        list: 각 PDF URL에서 추출한 텍스트와 메타데이터의 목록
    """
    results = []
    
    for url in urls:
        result = extract_text_from_pdf_url(url)
        if result:
            results.append(result)
    
    logger.info(f"총 {len(urls)}개 PDF URL 중 {len(results)}개에서 텍스트 추출 성공")
    return results

def process_pdf_files(file_paths):
    """
    여러 로컬 PDF 파일에서 텍스트 추출

    Args:
        file_paths (list): 텍스트를 추출할 PDF 파일 경로 목록

    Returns:
        list: 각 PDF 파일에서 추출한 텍스트와 메타데이터의 목록
    """
    results = []

    for file_path in file_paths:
        result = extract_text_from_pdf_file(file_path)
        if result:
            results.append(result)

    logger.info(f"총 {len(file_paths)}개 PDF 파일 중 {len(results)}개에서 텍스트 추출 성공")
    return results

if __name__ == "__main__":
    # 테스트 코드
    # 1. URL 테스트
    print("--- URL 테스트 시작 ---")
    test_urls = [
        "https://arxiv.org/pdf/2005.14165.pdf",  # 정상적인 PDF
        "https://www.example.com/nonexistent.pdf", # 존재하지 않는 PDF URL
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf" # 간단한 PDF
        # "https://www.example.com/password_protected.pdf" # 테스트용 암호화된 PDF (실제 URL 필요)
        # "https://www.example.com/corrupted.pdf" # 테스트용 손상된 PDF (실제 URL 필요)
    ]
    url_results = process_pdf_urls(test_urls)

    print(f"\n--- URL 테스트 결과 ({len(url_results)}/{len(test_urls)} 성공) ---")
    for res in url_results:
        print(f"URL: {res.get('url', res.get('file_path'))}")
        print(f"  파일명: {res['filename']}")
        print(f"  추출 방법: {res['method']}")
        print(f"  텍스트 길이: {res['length']}")
        print(f"  텍스트 일부: {res['text'][:100].replace('\n', ' ')}...\n")

    # 2. 로컬 파일 테스트 (테스트를 위해 임시 파일 생성 및 가짜 경로 사용)
    print("\n--- 로컬 파일 테스트 시작 ---")

    # 임시 정상 PDF 파일 생성 (PyPDF2를 사용하여 간단한 PDF 생성)
    # 실제 테스트 시에는 다양한 정상/문제 PDF 파일을 준비해야 합니다.
    temp_dir = tempfile.gettempdir()
    dummy_pdf_path = os.path.join(temp_dir, "dummy_correct.pdf")

    try:
        from PyPDF2 import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=210, height=297) # A4 size in points
        # You could add text here if PyPDF2 had an easy way to write text directly to a page.
        # For simplicity, we'll use a blank page. Actual text extraction won't yield much.
        with open(dummy_pdf_path, "wb") as f:
            writer.write(f)
        logger.info(f"임시 정상 PDF 파일 생성: {dummy_pdf_path}")
        created_dummy_pdf = True
    except Exception as e:
        logger.error(f"임시 PDF 생성 실패: {e}")
        created_dummy_pdf = False

    test_files = []
    if created_dummy_pdf:
        test_files.append(dummy_pdf_path)
    test_files.append("non_existent_file.pdf") # 존재하지 않는 파일

    # 테스트를 위해, 실제 손상된 파일이나 암호화된 파일을 이 리스트에 추가할 수 있습니다.
    # 예: test_files.append("path/to/your/corrupted.pdf")
    # 예: test_files.append("path/to/your/password_protected.pdf")

    file_results = process_pdf_files(test_files)

    print(f"\n--- 로컬 파일 테스트 결과 ({len(file_results)}/{len(test_files)} 성공) ---")
    for res in file_results:
        print(f"경로: {res.get('url', res.get('file_path'))}")
        print(f"  파일명: {res['filename']}")
        print(f"  추출 방법: {res['method']}")
        print(f"  텍스트 길이: {res['length']}")
        print(f"  텍스트 일부: {res['text'][:100].replace('\n', ' ')}...\n")

    # 임시 파일 삭제
    if created_dummy_pdf and os.path.exists(dummy_pdf_path):
        try:
            os.remove(dummy_pdf_path)
            logger.info(f"임시 PDF 파일 삭제: {dummy_pdf_path}")
        except Exception as e:
            logger.error(f"임시 PDF 파일 삭제 실패: {e}")