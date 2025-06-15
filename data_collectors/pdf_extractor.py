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
    
    except Exception as e:
        logger.error(f"PyPDF2로 텍스트 추출 중 오류 발생: {str(e)}")
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
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text
    
    except Exception as e:
        logger.error(f"pdfplumber로 텍스트 추출 중 오류 발생: {str(e)}")
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
    
    except Exception as e:
        logger.error(f"로컬 PDF 파일 처리 중 오류 발생: {str(e)}")
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

if __name__ == "__main__":
    # 테스트 코드
    test_url = "https://arxiv.org/pdf/2005.14165.pdf"  # 샘플 PDF URL
    result = extract_text_from_pdf_url(test_url)
    
    if result:
        print(f"파일명: {result['filename']}")
        print(f"추출 방법: {result['method']}")
        print(f"텍스트 길이: {result['length']}")
        print(f"텍스트 일부: {result['text'][:500]}...")
    else:
        print("텍스트 추출 실패")