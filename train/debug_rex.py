import json
import re
from pathlib import Path

# ==========================================
# 설정 (파일 경로 확인!)
# ==========================================
DATA_PATH = "train/train.jsonl" 

def check_regex_logic():
    print(f"=== 정규표현식(Regex) 데이터 추출 테스트: {DATA_PATH} ===")
    
    # 1. 파일이 있는지 확인
    if not Path(DATA_PATH).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    # 2. 정규표현식 정의 (학습 코드와 똑같이 설정)
    pat_type = re.compile(r"사고유형=.*?\(코드=(\d+)\)")
    pat_place = re.compile(r"사고장소=.*?\(코드=(\d+)\)")
    pat_feat  = re.compile(r"장소특징=.*?\(코드=(\d+)\)")
    pat_a     = re.compile(r"A차량.*?\(코드=(\d+)\)")
    pat_b     = re.compile(r"B차량.*?\(코드=(\d+)\)")

    # 3. 데이터 로드 및 검사
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"총 데이터 개수: {len(lines)}")
    print("-" * 50)

    # 4. 처음 5개 샘플만 확인
    success_count = 0
    
    for i in range(min(5, len(lines))):
        row = json.loads(lines[i])
        text = row["text"]
        
        # 추출 시도
        val_type = int(pat_type.search(text).group(1)) if pat_type.search(text) else 0
        val_place = int(pat_place.search(text).group(1)) if pat_place.search(text) else 0
        val_feat = int(pat_feat.search(text).group(1)) if pat_feat.search(text) else 0
        val_a = int(pat_a.search(text).group(1)) if pat_a.search(text) else 0
        val_b = int(pat_b.search(text).group(1)) if pat_b.search(text) else 0
        
        print(f"[Sample {i}]")
        print(f"Text: {text[:80]}...") # 텍스트 앞부분만 출력
        print(f"👉 결과: Type={val_type}, Place={val_place}, Feat={val_feat}, A={val_a}, B={val_b}")
        
        # 하나라도 0이 아니면 성공으로 간주
        if val_type != 0 or val_place != 0 or val_a != 0 or val_b != 0:
            success_count += 1
            print("✅ 추출 성공")
        else:
            print("❌ 추출 실패 (모두 0)")
        print("-" * 50)

    # 5. 종합 진단
    if success_count == 0:
        print("\n🚨 [심각] 모든 코드가 0으로 나옵니다!")
        print("원인: 정규표현식이 데이터 포맷과 맞지 않습니다.")
        print("해결: 위 출력된 'Text'를 복사해서 보여주시면 정규식을 수정해 드립니다.")
    else:
        print("\n🎉 [정상] 데이터 파싱은 문제 없습니다.")
        print("모델 학습이 안 되는 건 데이터 문제가 아니라 모델 파라미터(Learning Rate 등) 문제입니다.")

if __name__ == "__main__":
    check_regex_logic()