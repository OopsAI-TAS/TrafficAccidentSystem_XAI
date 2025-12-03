# check_raw_ig.py (새로 작성하거나 노트북에서 실행)

from ig_explain import load_model_and_tokenizer, integrated_gradients_embedding
import torch

# 1. 모델 로드 (작성하신 함수 활용)
tokenizer, model = load_model_and_tokenizer()

# 2. 테스트 문장 (가장 전형적인 예시 하나)
text = "[사고 정보] 사고유형=직진대좌회전, 장소=교차로내 [차량 진행] A차량=직진, B차량=좌회전" 
# (주의: 실제 데이터에 있는 포맷인 '코드=숫자' 형태로 넣으셔도 됩니다)

# 3. IG 계산 (작성하신 함수 그대로 사용)
print("Computing IG...")
# n_steps는 테스트니 20 정도로 낮춰서 빨리 확인
tokens, scores = integrated_gradients_embedding(model, tokenizer, text, n_steps=20) 

# 4. ★★★ 여기서 Merge 하지 말고 그대로 출력해보세요 ★★★
print(f"\n{'Token':<15} | {'Score':<10}")
print("-" * 30)

for t, s in zip(tokens, scores):
    # 특수문자나 0점 근처가 어떻게 나오는지 확인이 목적
    if t == "[PAD]": break
    print(f"{t:<15} | {s:.5f}")