#!/usr/bin/env python3
from datasets import load_dataset

configs_to_test = ['normal_instructions', 'tiny-textbooks', 'claude_evol']

for config in configs_to_test:
    try:
        print(f"\n=== 테스트 중: {config} ===")
        ds = load_dataset("maywell/korean_textbooks", config, split="train", trust_remote_code=True)
        print(f"데이터셋 크기: {len(ds)}")
        print(f"컬럼들: {ds.column_names}")
        print(f"첫 번째 샘플: {ds[0]}")
        print(f"샘플의 키들: {list(ds[0].keys())}")
        break  # 첫 번째 성공한 config를 사용
    except Exception as e:
        print(f"오류 발생 ({config}): {e}")
        continue