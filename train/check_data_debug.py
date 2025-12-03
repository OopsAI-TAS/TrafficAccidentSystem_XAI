import json
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from train.train_hybrid_parsing import HybridParsingDataset # 기존 학습 코드에서 클래스 임포트

# 설정
MODEL_NAME = "bert-base-multilingual-cased"
PATH = "train/train.jsonl" # 데이터 경로 확인

def check_parsing():
    print("=== 데이터 파싱 긴급 점검 ===")
    
    # 1. 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. 데이터셋 생성 (학습 때 쓴 클래스 그대로 사용)
    ds = HybridParsingDataset(PATH, tok)
    
    print(f"총 데이터 개수: {len(ds)}")
    
    # 3. 앞쪽 5개만 샘플링해서 실제로 뭐가 추출되는지 확인
    non_zero_count = 0
    
    for i in range(min(10, len(ds))):
        item = ds[i]
        raw_text = ds.rows[i]["text"]
        
        # 추출된 코드 값 확인
        c_type = item["c_type"].item()
        c_place = item["c_place"].item()
        c_a = item["c_a"].item()
        c_b = item["c_b"].item()
        
        print(f"\n[Sample {i}]")
        print(f"원본 텍스트 일부: {raw_text[:60]}...")
        print(f"👉 추출된 코드 | Type: {c_type}, Place: {c_place}, A: {c_a}, B: {c_b}")
        
        # 하나라도 0이 아니면 성공
        if c_type != 0 or c_place != 0 or c_a != 0 or c_b != 0:
            non_zero_count += 1

    print("-" * 30)
    if non_zero_count == 0:
        print("🚨 비상! 모든 코드가 0으로 잡히고 있습니다. 정규표현식이 틀렸습니다.")
        print("해결책: 데이터의 괄호나 띄어쓰기를 확인하고 Regex를 수정해야 합니다.")
    else:
        print(f"✅ 다행히 코드가 잡히고 있습니다. (Non-zero 샘플 수: {non_zero_count})")
        print("이 경우엔 모델 용량이나 하이퍼파라미터 문제일 수 있습니다.")

if __name__ == "__main__":
    check_parsing()