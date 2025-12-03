import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import re
from captum.attr import LayerIntegratedGradients

# ==========================================
# 1. 저장된 Hybrid 모델 구조 정의 (학습 코드와 동일해야 함)
# ==========================================
MODEL_NAME = "bert-base-multilingual-cased"
NUM_TYPE = 1000; NUM_PLACE = 500; NUM_FEAT = 500; NUM_MOVE = 200; EMBED_DIM = 16

class TrafficHybridRegressor(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.emb_type = nn.Embedding(NUM_TYPE, EMBED_DIM)
        self.emb_place = nn.Embedding(NUM_PLACE, EMBED_DIM)
        self.emb_feat = nn.Embedding(NUM_FEAT, EMBED_DIM)
        self.emb_a = nn.Embedding(NUM_MOVE, EMBED_DIM)
        self.emb_b = nn.Embedding(NUM_MOVE, EMBED_DIM)
        
        combined_dim = bert.config.hidden_size + (EMBED_DIM * 5)
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, c_type, c_place, c_feat, c_a, c_b):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]
        v_type = self.emb_type(c_type); v_place = self.emb_place(c_place)
        v_feat = self.emb_feat(c_feat); v_a = self.emb_a(c_a); v_b = self.emb_b(c_b)
        combined = torch.cat([h_cls, v_type, v_place, v_feat, v_a, v_b], dim=1)
        return self.regressor(combined).squeeze(-1)

# ==========================================
# 2. Wrapper (IG 분석용)
# ==========================================
class InterpretWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask, c_type, c_place, c_feat, c_a, c_b):
        return self.model(input_ids, attention_mask, c_type, c_place, c_feat, c_a, c_b)

# ==========================================
# 3. 실행 로직
# ==========================================
def extract_code(pattern, text, limit):
    m = pattern.search(text)
    if m:
        val = int(m.group(1))
        return val if val < limit else 0
    return 0

def main():
    CKPT = Path("train/artifacts_hybrid")
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    bert = AutoModel.from_pretrained(str(CKPT))
    model = TrafficHybridRegressor(bert)
    model.load_state_dict(torch.load(CKPT / "model.pt"))
    model.eval().cuda()

    # 테스트 문장 (학습 데이터와 같은 포맷)
    text = "[사고 정보] 사고유형=직진대좌회전(코드=214), 사고장소=교차로(코드=25) [차량 진행] A차량=직진(코드=6), B차량=좌회전(코드=14)"
    
    # 정규식 준비
    pat_type = re.compile(r"사고유형=.*?\(코드=(\d+)\)")
    pat_place = re.compile(r"사고장소=.*?\(코드=(\d+)\)")
    pat_feat = re.compile(r"장소특징=.*?\(코드=(\d+)\)")
    pat_a = re.compile(r"A차량.*?\(코드=(\d+)\)")
    pat_b = re.compile(r"B차량.*?\(코드=(\d+)\)")

    # 입력 준비
    enc = tok(text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
    input_ids = enc["input_ids"].cuda()
    mask = enc["attention_mask"].cuda()
    
    c_type = torch.tensor([extract_code(pat_type, text, NUM_TYPE)]).cuda()
    c_place = torch.tensor([extract_code(pat_place, text, NUM_PLACE)]).cuda()
    c_feat = torch.tensor([extract_code(pat_feat, text, NUM_FEAT)]).cuda()
    c_a = torch.tensor([extract_code(pat_a, text, NUM_MOVE)]).cuda()
    c_b = torch.tensor([extract_code(pat_b, text, NUM_MOVE)]).cuda()

    print(f"\n입력 텍스트: {text}")
    print(f"추출된 코드 -> Type:{c_type.item()}, Place:{c_place.item()}, A:{c_a.item()}, B:{c_b.item()}")
    
    # 예측값 확인
    with torch.no_grad():
        pred = model(input_ids, mask, c_type, c_place, c_feat, c_a, c_b)
    print(f"예측 과실비율: {pred.item():.2f}")

    # ==========================================
    # BERT Text에 대한 IG 계산
    # ==========================================
    wrapper = InterpretWrapper(model)
    lig = LayerIntegratedGradients(wrapper, model.bert.embeddings)
    
    # Baseline은 0 (Pad)
    attr, delta = lig.attribute(
        inputs=input_ids,
        baselines=torch.zeros_like(input_ids),
        additional_forward_args=(mask, c_type, c_place, c_feat, c_a, c_b),
        return_convergence_delta=True
    )
    
    # 결과 정리
    attr_score = attr.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    tokens = tok.convert_ids_to_tokens(input_ids[0])

    print("\n[BERT Text Importance (Top Tokens)]")
    print("-" * 40)
    for t, s in zip(tokens, attr_score):
        if t == "[PAD]": break
        if abs(s) > 0.05: # 의미 있는 점수만 출력
            print(f"{t:<15} | {s.item():.4f}")

if __name__ == "__main__":
    main()