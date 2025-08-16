#!/usr/bin/env python3
from datasets import load_dataset

try:
    print("데이터셋 로딩 테스트 중...")
    ds = load_dataset("maywell/korean_textbooks", split="train", trust_remote_code=True)
    print(f"데이터셋 크기: {len(ds)}")
    print(f"데이터셋 컬럼: {ds.column_names}")
    print(f"첫 번째 샘플: {ds[0]}")
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()