"""
웹페이지에서 텍스트를 추출하는 모듈
"""
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import logging
import time
from urllib.parse import urljoin, urlparse
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('web_scraper')

def extract_text_with_bs4(url):
    """
    BeautifulSoup을 사용하여 웹페이지에서 텍스트 추출
    
    Args:
        url (str): 텍스트를 추출할 웹페이지 URL
        
    Returns:
        str: 추출된 텍스트
    """
    try:
        # 웹페이지 요청
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        
        # HTML 파싱
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 불필요한 태그 제거
        for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
            tag.decompose()
        
        # 텍스트 추출 및 정리
        text = soup.get_text(separator=' ')
        lines = [line.strip() for line in text.split('\n')]
        text = ' '.join(line for line in lines if line)
        
        return text
    
    except Exception as e:
        logger.error(f"BeautifulSoup으로 텍스트 추출 중 오류 발생: {str(e)}")
        return None

def extract_text_with_newspaper(url):
    """
    Newspaper3k를 사용하여 웹페이지에서 텍스트 추출 (기사에 최적화)
    
    Args:
        url (str): 텍스트를 추출할 웹페이지 URL
        
    Returns:
        dict: 제목, 본문, 요약, 키워드 등을 포함한 추출 결과
    """
    try:
        # 기사 객체 생성 및 다운로드
        article = Article(url, language='ko')  # 한국어 설정
        article.download()
        article.parse()
        
        # 자연어 처리 (NLP) 실행
        try:
            article.nlp()
        except Exception as nlp_error:
            logger.warning(f"NLP 처리 중 오류 발생 (계속 진행): {str(nlp_error)}")
        
        # 추출된 정보 반환
        result = {
            'title': article.title,
            'text': article.text,
            'summary': getattr(article, 'summary', ''),
            'keywords': getattr(article, 'keywords', []),
            'publish_date': str(article.publish_date) if article.publish_date else None,
            'url': url
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Newspaper3k로 텍스트 추출 중 오류 발생: {str(e)}")
        return None

def extract_text_from_url(url, method='auto'):
    """
    웹페이지에서 텍스트 추출 (적절한 방법 자동 선택 또는 지정)
    
    Args:
        url (str): 텍스트를 추출할 웹페이지 URL
        method (str): 추출 방법 ('bs4', 'newspaper', 'auto')
        
    Returns:
        dict: 추출된 텍스트와 메타데이터
    """
    logger.info(f"URL에서 텍스트 추출 시작: {url}")
    
    if method == 'bs4':
        text = extract_text_with_bs4(url)
        if text:
            return {'text': text, 'url': url, 'method': 'bs4'}
        return None
    
    elif method == 'newspaper':
        article_data = extract_text_with_newspaper(url)
        if article_data:
            article_data['method'] = 'newspaper'
            return article_data
        return None
    
    else:  # 'auto' 방식
        # 먼저 Newspaper3k 시도
        article_data = extract_text_with_newspaper(url)
        if article_data and article_data['text']:
            article_data['method'] = 'newspaper'
            return article_data
        
        # Newspaper3k 실패 시 BeautifulSoup 시도
        text = extract_text_with_bs4(url)
        if text:
            return {'text': text, 'url': url, 'method': 'bs4'}
        
        # 모두 실패한 경우
        logger.error(f"URL에서 텍스트 추출 실패: {url}")
        return None

def process_urls(urls):
    """
    여러 URL에서 텍스트 추출
    
    Args:
        urls (list): 텍스트를 추출할 URL 목록
        
    Returns:
        list: 각 URL에서 추출한 텍스트와 메타데이터의 목록
    """
    results = []
    
    for url in urls:
        result = extract_text_from_url(url)
        if result:
            results.append(result)
    
    logger.info(f"총 {len(urls)}개 URL 중 {len(results)}개에서 텍스트 추출 성공")
    return results

def extract_links_from_page(url, max_links=50):
    """
    웹페이지에서 링크들을 추출
    
    Args:
        url (str): 링크를 추출할 웹페이지 URL
        max_links (int): 추출할 최대 링크 수
        
    Returns:
        list: 추출된 링크 URL 목록
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 기본 URL 정보
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        
        links = set()  # 중복 제거를 위해 set 사용
        
        # 모든 a 태그에서 href 추출
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # 상대 URL을 절대 URL로 변환
            absolute_url = urljoin(url, href)
            
            # 유효한 URL인지 확인
            if is_valid_url(absolute_url, base_url):
                links.add(absolute_url)
                
                # 최대 링크 수 제한
                if len(links) >= max_links:
                    break
        
        logger.info(f"{url}에서 {len(links)}개 링크 추출")
        return list(links)
    
    except Exception as e:
        logger.error(f"링크 추출 중 오류 발생: {str(e)}")
        return []

def is_valid_url(url, base_url):
    """
    URL이 크롤링에 적합한지 확인
    
    Args:
        url (str): 확인할 URL
        base_url (str): 기본 URL (도메인)
        
    Returns:
        bool: 유효한 URL인지 여부
    """
    try:
        parsed = urlparse(url)
        
        # HTTP/HTTPS만 허용
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # 같은 도메인만 허용 (내부 링크만)
        if not url.startswith(base_url):
            return False
        
        # 제외할 파일 확장자
        excluded_extensions = ['.pdf', '.jpg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.doc', '.docx', '.xls', '.xlsx']
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        # 제외할 패턴
        excluded_patterns = [
            r'#',  # 앵커 링크
            r'javascript:',  # 자바스크립트
            r'mailto:',  # 이메일
            r'/search',  # 검색 페이지
            r'/login',  # 로그인 페이지
            r'/register',  # 회원가입 페이지
        ]
        
        for pattern in excluded_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        return True
    
    except Exception:
        return False

def crawl_website_depth_1(base_url, max_pages=20, delay=1):
    """
    웹사이트를 1DEPTH까지 크롤링하여 텍스트 추출
    
    Args:
        base_url (str): 시작 URL
        max_pages (int): 최대 크롤링할 페이지 수
        delay (int): 페이지 간 지연 시간 (초)
        
    Returns:
        list: 추출된 텍스트와 메타데이터의 목록
    """
    logger.info(f"1DEPTH 크롤링 시작: {base_url}")
    
    results = []
    
    # 1. 기본 페이지에서 텍스트 추출
    base_result = extract_text_from_url(base_url)
    if base_result:
        base_result['depth'] = 0
        base_result['source_url'] = base_url
        results.append(base_result)
        logger.info(f"기본 페이지 처리 완료: {base_url}")
    
    # 2. 기본 페이지에서 링크 추출
    links = extract_links_from_page(base_url, max_links=max_pages)
    
    # 3. 각 링크에서 텍스트 추출
    processed_count = 0
    for i, link in enumerate(links):
        if processed_count >= max_pages:
            break
        
        try:
            logger.info(f"처리 중 ({i+1}/{len(links)}): {link}")
            
            # 지연 시간 적용
            if delay > 0:
                time.sleep(delay)
            
            # 텍스트 추출
            result = extract_text_from_url(link)
            if result and result.get('text'):
                result['depth'] = 1
                result['source_url'] = base_url
                result['parent_url'] = base_url
                results.append(result)
                processed_count += 1
                logger.info(f"성공: {link} ({len(result['text'])} 글자)")
            else:
                logger.warning(f"텍스트 추출 실패: {link}")
                
        except Exception as e:
            logger.error(f"링크 처리 중 오류: {link} - {str(e)}")
            continue
    
    logger.info(f"1DEPTH 크롤링 완료: 총 {len(results)}개 페이지 처리")
    return results

if __name__ == "__main__":
    # 테스트 코드
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "crawl":
        # 1DEPTH 크롤링 테스트
        test_url = "https://news.hada.io/show"
        results = crawl_website_depth_1(test_url, max_pages=5, delay=2)
        
        print(f"\n=== 크롤링 결과 ===\n")
        for i, result in enumerate(results):
            print(f"{i+1}. URL: {result.get('url', 'N/A')}")
            print(f"   Depth: {result.get('depth', 'N/A')}")
            print(f"   제목: {result.get('title', '제목 없음')}")
            print(f"   텍스트 길이: {len(result.get('text', ''))}")
            print(f"   텍스트 일부: {result.get('text', '')[:100]}...\n")
    else:
        # 기본 테스트
        test_url = "https://ko.wikipedia.org/wiki/인공지능"
        result = extract_text_from_url(test_url)
        
        if result:
            print(f"제목: {result.get('title', '제목 없음')}")
            print(f"텍스트 길이: {len(result['text'])}")
            print(f"텍스트 일부: {result['text'][:500]}...")
        else:
            print("텍스트 추출 실패")