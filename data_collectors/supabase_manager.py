"""
Supabase와 연동하여 학습 데이터를 관리하는 모듈
"""
import os
import json
import logging
from datetime import datetime
from supabase import create_client, Client

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('supabase_manager')

class SupabaseManager:
    """Supabase 데이터베이스 관리 클래스"""
    
    def __init__(self, url=None, key=None):
        """
        Supabase 클라이언트 초기화
        
        Args:
            url (str, optional): Supabase URL. 없으면 환경 변수에서 가져옴
            key (str, optional): Supabase API 키. 없으면 환경 변수에서 가져옴
        """
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        
        if not self.url or not self.key:
            logger.warning("Supabase URL 또는 API 키가 제공되지 않았습니다. 환경 변수 SUPABASE_URL과 SUPABASE_KEY를 설정하세요.")
            self.client = None
        else:
            try:
                self.client = create_client(self.url, self.key)
                logger.info("Supabase 클라이언트 초기화 성공")
            except Exception as e:
                logger.error(f"Supabase 클라이언트 초기화 오류: {str(e)}")
                self.client = None
    
    def set_credentials(self, url, key):
        """
        Supabase 자격 증명 설정
        
        Args:
            url (str): Supabase URL
            key (str): Supabase API 키
            
        Returns:
            bool: 성공 여부
        """
        self.url = url
        self.key = key
        
        try:
            self.client = create_client(self.url, self.key)
            logger.info("Supabase 자격 증명 설정 성공")
            return True
        except Exception as e:
            logger.error(f"Supabase 자격 증명 설정 오류: {str(e)}")
            self.client = None
            return False
    
    def create_training_data_table(self, table_name="training_data"):
        """
        학습 데이터를 저장할 테이블 생성
        
        Args:
            table_name (str): 생성할 테이블 이름
            
        Returns:
            bool: 성공 여부
        """
        if not self.client:
            logger.error("Supabase 클라이언트가 초기화되지 않았습니다.")
            return False
        
        try:
            # SQL 쿼리 작성 - PostgreSQL 문법 사용
            query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                text TEXT NOT NULL,
                source_type VARCHAR(50) NOT NULL,
                source_url TEXT,
                source_file TEXT,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            
            # 테이블 생성 실행
            self.client.table(table_name).query(query).execute()
            logger.info(f"테이블 '{table_name}' 생성 성공")
            return True
            
        except Exception as e:
            logger.error(f"테이블 생성 오류: {str(e)}")
            return False
    
    def insert_data(self, text, source_type, source_url=None, source_file=None, metadata=None, table_name="training_data"):
        """
        학습 데이터 삽입
        
        Args:
            text (str): 학습에 사용할 텍스트
            source_type (str): 데이터 소스 유형 ('web', 'pdf', 'text')
            source_url (str, optional): 소스 URL
            source_file (str, optional): 소스 파일 이름
            metadata (dict, optional): 추가 메타데이터
            table_name (str): 테이블 이름
            
        Returns:
            dict: 삽입된 데이터 또는 None (실패 시)
        """
        if not self.client:
            logger.error("Supabase 클라이언트가 초기화되지 않았습니다.")
            return None
        
        try:
            data = {
                "text": text,
                "source_type": source_type,
                "source_url": source_url,
                "source_file": source_file,
                "metadata": json.dumps(metadata) if metadata else None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # 데이터 삽입
            result = self.client.table(table_name).insert(data).execute()
            
            # 응답에서 삽입된 데이터 추출
            if result.data and len(result.data) > 0:
                logger.info(f"데이터 삽입 성공: ID {result.data[0].get('id')}")
                return result.data[0]
            else:
                logger.warning("데이터 삽입 실패: 응답에 데이터가 없음")
                return None
            
        except Exception as e:
            logger.error(f"데이터 삽입 오류: {str(e)}")
            return None
    
    def insert_web_data(self, web_data, table_name="training_data"):
        """
        웹 스크래핑 데이터 삽입
        
        Args:
            web_data (dict): 웹 스크래핑 결과 데이터
            table_name (str): 테이블 이름
            
        Returns:
            dict: 삽입된 데이터 또는 None (실패 시)
        """
        if not web_data or 'text' not in web_data:
            logger.error("유효하지 않은 웹 데이터")
            return None
        
        # 메타데이터 준비
        metadata = {k: v for k, v in web_data.items() if k not in ['text', 'url']}
        
        return self.insert_data(
            text=web_data['text'],
            source_type='web',
            source_url=web_data.get('url'),
            metadata=metadata,
            table_name=table_name
        )
    
    def insert_pdf_data(self, pdf_data, table_name="training_data"):
        """
        PDF 추출 데이터 삽입
        
        Args:
            pdf_data (dict): PDF 추출 결과 데이터
            table_name (str): 테이블 이름
            
        Returns:
            dict: 삽입된 데이터 또는 None (실패 시)
        """
        if not pdf_data or 'text' not in pdf_data:
            logger.error("유효하지 않은 PDF 데이터")
            return None
        
        # 메타데이터 준비
        metadata = {k: v for k, v in pdf_data.items() if k not in ['text', 'url', 'filename']}
        
        return self.insert_data(
            text=pdf_data['text'],
            source_type='pdf',
            source_url=pdf_data.get('url'),
            source_file=pdf_data.get('filename'),
            metadata=metadata,
            table_name=table_name
        )
    
    def insert_text_data(self, text, file_name=None, metadata=None, table_name="training_data"):
        """
        텍스트 파일 데이터 삽입
        
        Args:
            text (str): 삽입할 텍스트
            file_name (str, optional): 원본 파일 이름
            metadata (dict, optional): 추가 메타데이터
            table_name (str): 테이블 이름
            
        Returns:
            dict: 삽입된 데이터 또는 None (실패 시)
        """
        return self.insert_data(
            text=text,
            source_type='text',
            source_file=file_name,
            metadata=metadata,
            table_name=table_name
        )
    
    def get_all_data(self, table_name="training_data", limit=100, offset=0):
        """
        저장된 모든 학습 데이터 가져오기
        
        Args:
            table_name (str): 테이블 이름
            limit (int): 가져올 최대 데이터 수
            offset (int): 오프셋 (페이지네이션용)
            
        Returns:
            list: 학습 데이터 목록 또는 None (실패 시)
        """
        if not self.client:
            logger.error("Supabase 클라이언트가 초기화되지 않았습니다.")
            return None
        
        try:
            result = self.client.table(table_name).select("*").limit(limit).offset(offset).execute()
            
            if result.data:
                logger.info(f"{len(result.data)}개 데이터 조회 성공")
                return result.data
            else:
                logger.warning("조회된 데이터 없음")
                return []
            
        except Exception as e:
            logger.error(f"데이터 조회 오류: {str(e)}")
            return None
    
    def get_data_by_source_type(self, source_type, table_name="training_data", limit=100):
        """
        소스 유형별 학습 데이터 가져오기
        
        Args:
            source_type (str): 데이터 소스 유형 ('web', 'pdf', 'text')
            table_name (str): 테이블 이름
            limit (int): 가져올 최대 데이터 수
            
        Returns:
            list: 학습 데이터 목록 또는 None (실패 시)
        """
        if not self.client:
            logger.error("Supabase 클라이언트가 초기화되지 않았습니다.")
            return None
        
        try:
            result = self.client.table(table_name).select("*").eq("source_type", source_type).limit(limit).execute()
            
            if result.data:
                logger.info(f"{source_type} 유형 {len(result.data)}개 데이터 조회 성공")
                return result.data
            else:
                logger.warning(f"{source_type} 유형 데이터 없음")
                return []
            
        except Exception as e:
            logger.error(f"데이터 조회 오류: {str(e)}")
            return None
    
    def export_data_to_jsonl(self, output_file, table_name="training_data"):
        """
        학습 데이터를 JSONL 형식으로 내보내기
        
        Args:
            output_file (str): 출력 파일 경로
            table_name (str): 테이블 이름
            
        Returns:
            bool: 성공 여부
        """
        data = self.get_all_data(table_name, limit=10000)
        
        if data is None:
            logger.error("데이터를 가져올 수 없습니다.")
            return False
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    # JSON 직렬화하여 한 줄로 저장
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"{len(data)}개 데이터를 {output_file}에 저장했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"데이터 내보내기 오류: {str(e)}")
            return False
    
    def delete_data(self, data_id, table_name="training_data"):
        """
        특정 데이터 삭제
        
        Args:
            data_id (str): 삭제할 데이터 ID
            table_name (str): 테이블 이름
            
        Returns:
            bool: 성공 여부
        """
        if not self.client:
            logger.error("Supabase 클라이언트가 초기화되지 않았습니다.")
            return False
        
        try:
            result = self.client.table(table_name).delete().eq("id", data_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"ID {data_id} 데이터 삭제 성공")
                return True
            else:
                logger.warning(f"ID {data_id} 데이터 삭제 실패: 데이터를 찾을 수 없음")
                return False
            
        except Exception as e:
            logger.error(f"데이터 삭제 오류: {str(e)}")
            return False

if __name__ == "__main__":
    # 테스트 코드
    # 환경 변수 설정 또는 직접 자격 증명 입력 필요
    manager = SupabaseManager()
    
    if manager.client:
        # 테이블 생성
        manager.create_training_data_table()
        
        # 샘플 데이터 삽입
        sample_data = manager.insert_text_data(
            text="이것은 테스트 데이터입니다. 한국어 언어 모델 파인튜닝을 위한 샘플입니다.",
            file_name="sample.txt",
            metadata={"purpose": "testing", "language": "ko"}
        )
        
        if sample_data:
            print(f"샘플 데이터 삽입됨: {sample_data['id']}")
            
            # 데이터 조회
            all_data = manager.get_all_data(limit=5)
            if all_data:
                print(f"총 {len(all_data)}개 데이터 조회됨")
                
                # 첫 번째 데이터 출력
                print(json.dumps(all_data[0], ensure_ascii=False, indent=2))
    else:
        print("Supabase 연결에 실패했습니다. 자격 증명을 확인하세요.")