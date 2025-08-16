#!/usr/bin/env python3
"""
커스터마이즈 가능한 한국어 위키피디아 스크래퍼
다양한 시작 페이지와 카테고리에서 데이터 수집 가능
"""

import requests
import json
import time
import re
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import List, Dict, Set
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class CustomWikipediaScraper:
    def __init__(self, max_articles=100, delay=1.0, max_workers=3, start_url=None):
        """
        커스터마이즈 가능한 위키피디아 스크래퍼 초기화
        
        Args:
            max_articles: 수집할 최대 문서 수
            delay: 요청 간 지연 시간 (초)
            max_workers: 동시 작업자 수
            start_url: 시작 URL (None이면 기본 대문 사용)
        """
        self.max_articles = max_articles
        self.delay = delay
        self.max_workers = max_workers
        self.base_url = "https://ko.wikipedia.org"
        self.start_url = start_url or f"{self.base_url}/wiki/위키백과:대문"
        self.visited_urls: Set[str] = set()
        self.collected_articles: List[Dict] = []
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper_custom.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Linux; x86_64) Korean-LLM-Scraper/1.0'
        })
    
    def get_links_from_page(self, start_url: str) -> List[str]:
        """지정된 페이지에서 위키 링크 수집"""
        try:
            self.logger.info(f"페이지에서 링크 수집 중: {start_url}")
            response = self.session.get(start_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            # 메인 콘텐츠 영역 찾기
            main_content = soup.find('div', {'id': 'mw-content-text'})
            if not main_content:
                self.logger.warning("메인 콘텐츠를 찾을 수 없습니다")
                return []
            
            # 위키 링크만 필터링
            for link in main_content.find_all('a', href=True):
                href = link['href']
                # 일반 위키 문서 링크만 선택 (파일, 분류, 특수 페이지 제외)
                if (href.startswith('/wiki/') and 
                    ':' not in href and 
                    not href.startswith('/wiki/파일:') and
                    not href.startswith('/wiki/분류:') and
                    not href.startswith('/wiki/위키백과:')):
                    
                    full_url = urljoin(self.base_url, href)
                    if full_url not in self.visited_urls:
                        links.append(full_url)
            
            self.logger.info(f"페이지에서 {len(links)}개 링크 발견")
            return links[:self.max_articles]
            
        except Exception as e:
            self.logger.error(f"페이지 링크 수집 실패: {e}")
            return []
    
    def get_category_links(self, category_name: str) -> List[str]:
        """특정 카테고리에서 문서 링크 수집"""
        category_url = f"{self.base_url}/wiki/분류:{category_name}"
        
        try:
            self.logger.info(f"카테고리에서 링크 수집 중: {category_name}")
            response = self.session.get(category_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            # 카테고리 멤버 찾기
            category_members = soup.find('div', {'id': 'mw-pages'})
            if not category_members:
                self.logger.warning("카테고리 멤버를 찾을 수 없습니다")
                return []
            
            for link in category_members.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = urljoin(self.base_url, href)
                    if full_url not in self.visited_urls:
                        links.append(full_url)
            
            self.logger.info(f"카테고리에서 {len(links)}개 링크 발견")
            return links[:self.max_articles]
            
        except Exception as e:
            self.logger.error(f"카테고리 링크 수집 실패: {e}")
            return []
    
    def search_wikipedia(self, search_term: str, limit: int = None) -> List[str]:
        """위키피디아 검색 결과에서 링크 수집"""
        search_url = f"{self.base_url}/w/api.php"
        
        try:
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': search_term,
                'srlimit': limit or self.max_articles,
                'format': 'json'
            }
            
            self.logger.info(f"검색어로 링크 수집 중: {search_term}")
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            links = []
            
            if 'query' in data and 'search' in data['query']:
                for result in data['query']['search']:
                    title = result['title']
                    # 제목을 URL로 변환
                    url = f"{self.base_url}/wiki/{title.replace(' ', '_')}"
                    if url not in self.visited_urls:
                        links.append(url)
            
            self.logger.info(f"검색에서 {len(links)}개 링크 발견")
            return links
            
        except Exception as e:
            self.logger.error(f"검색 실패: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제 함수"""
        if not text:
            return ""
        
        # HTML 엔티티 처리
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        
        # 연속된 공백 및 줄바꿈 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 특수 위키 문법 제거
        text = re.sub(r'\[\[파일:.*?\]\]', '', text)
        text = re.sub(r'\[\[분류:.*?\]\]', '', text)
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
        
        # 편집 링크 제거
        text = re.sub(r'\[편집\]', '', text)
        text = re.sub(r'\[원문 편집\]', '', text)
        
        # 참조 번호 제거
        text = re.sub(r'\[\d+\]', '', text)
        
        return text.strip()
    
    def extract_article_content(self, url: str) -> Dict[str, str]:
        """개별 문서에서 본문 내용 추출"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 제목 추출
            title_elem = soup.find('h1', {'class': 'firstHeading'})
            title = title_elem.get_text().strip() if title_elem else "제목 없음"
            
            # 본문 내용 추출
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return {"title": title, "content": "", "url": url}
            
            # 불필요한 요소 제거
            for element in content_div.find_all(['table', 'div'], 
                                               {'class': ['navbox', 'infobox', 'toc', 'navbox-group']}):
                element.decompose()
            
            # 참조 섹션 제거
            for element in content_div.find_all(['div', 'span'], 
                                               {'class': ['reflist', 'references']}):
                element.decompose()
            
            # 텍스트 추출 및 정제
            paragraphs = content_div.find_all('p')
            content_parts = []
            
            for p in paragraphs:
                text = p.get_text()
                cleaned = self.clean_text(text)
                if len(cleaned) > 50:  # 너무 짧은 문단 제외
                    content_parts.append(cleaned)
            
            content = '\n\n'.join(content_parts)
            
            # 최소 길이 확인
            if len(content) < 200:
                self.logger.warning(f"내용이 너무 짧음: {title} ({len(content)} chars)")
                return {"title": title, "content": "", "url": url}
            
            self.logger.info(f"수집 완료: {title} ({len(content)} chars)")
            return {
                "title": title,
                "content": content,
                "url": url,
                "length": len(content)
            }
            
        except Exception as e:
            self.logger.error(f"문서 수집 실패 {url}: {e}")
            return {"title": "오류", "content": "", "url": url}
    
    def scrape_article(self, url: str) -> Dict[str, str]:
        """단일 문서 스크래핑 (스레드용)"""
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        time.sleep(self.delay)  # 요청 제한
        
        return self.extract_article_content(url)
    
    def collect_articles(self, urls: List[str]) -> List[Dict]:
        """병렬로 문서들 수집"""
        valid_articles = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            future_to_url = {
                executor.submit(self.scrape_article, url): url 
                for url in urls[:self.max_articles]
            }
            
            # 결과 수집
            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                completed += 1
                
                try:
                    article = future.result()
                    if article and article['content']:
                        valid_articles.append(article)
                        self.logger.info(f"진행률: {completed}/{len(future_to_url)} - "
                                       f"유효 문서: {len(valid_articles)}")
                except Exception as e:
                    self.logger.error(f"문서 처리 실패 {url}: {e}")
        
        return valid_articles
    
    def save_to_jsonl(self, articles: List[Dict], filename: str = "korean_wikipedia_data.jsonl"):
        """수집된 데이터를 JSONL 형식으로 저장"""
        output_path = Path(filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for article in articles:
                    # 학습 데이터 형식으로 변환
                    training_text = f"제목: {article['title']}\n\n{article['content']}"
                    
                    json_line = {
                        "text": training_text,
                        "title": article['title'],
                        "url": article['url'],
                        "length": len(training_text)
                    }
                    
                    f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
            
            self.logger.info(f"데이터 저장 완료: {output_path} ({len(articles)} 문서)")
            
            # 통계 정보 출력
            total_chars = sum(article['length'] for article in articles)
            avg_length = total_chars / len(articles) if articles else 0
            
            print(f"\n📊 수집 통계:")
            print(f"   총 문서 수: {len(articles)}")
            print(f"   총 문자 수: {total_chars:,}")
            print(f"   평균 길이: {avg_length:.0f} 문자")
            print(f"   저장 파일: {output_path}")
            
        except Exception as e:
            self.logger.error(f"파일 저장 실패: {e}")
    
    def run_with_custom_source(self, source_type: str, source_value: str, output_file: str):
        """커스텀 소스로 스크래핑 실행"""
        self.logger.info(f"커스텀 소스로 데이터 수집 시작: {source_type} = {source_value}")
        
        # 소스 타입에 따라 링크 수집
        if source_type == "url":
            print(f"🔍 지정된 URL에서 링크 수집 중: {source_value}")
            links = self.get_links_from_page(source_value)
        elif source_type == "category":
            print(f"🔍 카테고리에서 링크 수집 중: {source_value}")
            links = self.get_category_links(source_value)
        elif source_type == "search":
            print(f"🔍 검색어로 링크 수집 중: {source_value}")
            links = self.search_wikipedia(source_value)
        else:
            self.logger.error(f"지원되지 않는 소스 타입: {source_type}")
            return False
        
        if not links:
            self.logger.error("링크 수집 실패")
            return False
        
        # 문서 내용 수집
        print(f"📚 {len(links)}개 문서 수집 중...")
        articles = self.collect_articles(links)
        
        if not articles:
            self.logger.error("문서 수집 실패")
            return False
        
        # JSONL 파일로 저장
        print("💾 데이터 저장 중...")
        self.save_to_jsonl(articles, output_file)
        
        print("✅ 데이터 수집 완료!")
        return True

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="커스터마이즈 가능한 한국어 위키피디아 데이터 수집")
    parser.add_argument("--max-articles", type=int, default=50,
                       help="수집할 최대 문서 수 (기본값: 50)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="요청 간 지연 시간 초 (기본값: 1.0)")
    parser.add_argument("--workers", type=int, default=3,
                       help="동시 작업자 수 (기본값: 3)")
    parser.add_argument("--output", type=str, default="korean_wikipedia_data.jsonl",
                       help="출력 파일명")
    
    # 커스텀 소스 옵션들
    parser.add_argument("--source-type", type=str, choices=["url", "category", "search"],
                       help="데이터 소스 타입: url(특정 페이지), category(카테고리), search(검색어)")
    parser.add_argument("--source-value", type=str,
                       help="소스 값 (URL, 카테고리명, 또는 검색어)")
    
    args = parser.parse_args()
    
    # 데이터 디렉토리 생성
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / args.output
    
    # 스크래퍼 실행
    scraper = CustomWikipediaScraper(
        max_articles=args.max_articles,
        delay=args.delay,
        max_workers=args.workers
    )
    
    if args.source_type and args.source_value:
        # 커스텀 소스 사용
        success = scraper.run_with_custom_source(
            args.source_type, 
            args.source_value, 
            str(output_path)
        )
    else:
        # 기본 대문 사용
        print("🔍 위키피디아 대문에서 링크 수집 중...")
        links = scraper.get_links_from_page(scraper.start_url)
        
        if not links:
            exit(1)
        
        print(f"📚 {len(links)}개 문서 수집 중...")
        articles = scraper.collect_articles(links)
        
        if not articles:
            exit(1)
        
        print("💾 데이터 저장 중...")
        scraper.save_to_jsonl(articles, str(output_path))
        
        print("✅ 데이터 수집 완료!")
        success = True
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()