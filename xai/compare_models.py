import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import re
from captum.attr import LayerIntegratedGradients

# ==============================================================================
# 1. 설정 및 모델 정의 (두 모델의 구조를 모두 정의해야 함)
# ==============================================================================
MODEL_NAME = "bert-base-multilingual-cased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- [모델 A] 기존 텍스트 전용 모델 (TextOnlyHead) ---
class TextOnlyHead(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.head = nn.Linear(bert.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        # Captum 배치 대응용 확장
        if input_ids is not None and attention_mask is not None:
            if attention_mask.shape[0] != input_ids.shape[0]:
                attention_mask = attention_mask.expand(input_ids.shape[0], -1)
                
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]
        logits = self.head(h_cls)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 0] * 100

# --- [모델 B] 신규 하이브리드 모델 (TrafficHybridSoftmax) ---
NUM_TYPE = 1000; NUM_PLACE = 500; NUM_FEAT = 500; NUM_MOVE = 300; EMBED_DIM = 32
class TrafficHybridSoftmax(nn.Module):
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
        self.classifier = nn.Linear(combined_dim, 2) 

    def forward(self, input_ids, attention_mask, c_type, c_place, c_feat, c_a, c_b):
        if input_ids is not None and attention_mask is not None:
            if attention_mask.shape[0] != input_ids.shape[0]:
                attention_mask = attention_mask.expand(input_ids.shape[0], -1)

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]
        v_type = self.emb_type(c_type); v_place = self.emb_place(c_place)
        v_feat = self.emb_feat(c_feat); v_a = self.emb_a(c_a); v_b = self.emb_b(c_b)
        combined = torch.cat([h_cls, v_type, v_place, v_feat, v_a, v_b], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 0] * 100

# ==============================================================================
# 2. 유틸리티 (파싱 등)
# ==============================================================================
def extract_code(pattern, text, max_limit):
    m = pattern.search(text)
    if m:
        val = int(m.group(1))
        return val if val < max_limit else 0
    return 0

def load_old_model(path):
    # BERT는 공유하되 Head만 다름
    bert = AutoModel.from_pretrained(MODEL_NAME)
    model = TextOnlyHead(bert)
    try:
        model.load_state_dict(torch.load(path))
    except:
        print("⚠️ 기존 모델 가중치 로드 실패 (구조 불일치 등).")
        return None
    return model.to(DEVICE).eval()

def load_new_model(path):
    bert = AutoModel.from_pretrained(MODEL_NAME)
    model = TrafficHybridSoftmax(bert)
    try:
        model.load_state_dict(torch.load(path))
    except:
        print("⚠️ 신규 모델 가중치 로드 실패.")
        return None
    return model.to(DEVICE).eval()

# ==============================================================================
# 3. 메인 비교 로직
# ==============================================================================
def main():
    # 경로 설정 (사용자 환경에 맞게 수정)
    PATH_OLD = Path("train/artifacts/model.pt")
    PATH_NEW = Path("train/artifacts_hybrid_softmax/best_model.pt")
    
    print("=== ⚔️ Model XAI Battle: Old vs New ⚔️ ===")
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 모델 로드
    print("1. Loading Old Model (Text Only)...")
    model_old = load_old_model(PATH_OLD)
    
    print("2. Loading New Model (Hybrid)...")
    model_new = load_new_model(PATH_NEW)
    
    if model_old is None or model_new is None:
        print("❌ 모델 로드 실패로 종료합니다.")
        return

    # ★ 비교할 테스트 문장 (가장 전형적인 케이스)
    text = "[사고 정보] 사고유형=직진대좌회전(코드=214), 사고장소=교차로(코드=25) [차량 진행] A차량=직진(코드=6), B차량=좌회전(코드=14)"
    
    print(f"\n[Input Text]\n{text}")
    print("-" * 60)

    # ---------------------------------------------------------
    # [분석 1] Old Model (Text Only)
    # ---------------------------------------------------------
    enc = tok(text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    
    print("\n🕵️‍♂️ [Old Model Analysis]")
    with torch.no_grad():
        pred_old = model_old(ids, mask).item()
    print(f"👉 예측값: {pred_old:.2f} %")
    
    lig_old = LayerIntegratedGradients(model_old, model_old.bert.embeddings)
    attr_old = lig_old.attribute(inputs=ids, additional_forward_args=(mask,), n_steps=50)
    score_old = attr_old.sum(dim=2).squeeze(0)
    score_old = score_old / torch.norm(score_old)
    
    # ---------------------------------------------------------
    # [분석 2] New Model (Hybrid)
    # ---------------------------------------------------------
    # 정규식 파싱
    pat_type = re.compile(r"사고유형=[^,]*?\(?(?:코드=)?(\d+)\)?")
    pat_place = re.compile(r"사고장소=[^,]*?\(?(?:코드=)?(\d+)\)?")
    pat_feat  = re.compile(r"장소특징=[^,]*?\(?(?:코드=)?(\d+)\)?")
    pat_a     = re.compile(r"A차량[^,]*?\(?(?:코드=)?(\d+)\)?")
    pat_b     = re.compile(r"B차량[^,]*?\(?(?:코드=)?(\d+)\)?")
    
    c_type = torch.tensor([extract_code(pat_type, text, 1000)]).to(DEVICE)
    c_place = torch.tensor([extract_code(pat_place, text, 500)]).to(DEVICE)
    c_feat = torch.tensor([extract_code(pat_feat, text, 500)]).to(DEVICE)
    c_a = torch.tensor([extract_code(pat_a, text, 300)]).to(DEVICE)
    c_b = torch.tensor([extract_code(pat_b, text, 300)]).to(DEVICE)
    
    print("\n🕵️‍♂️ [New Model Analysis]")
    with torch.no_grad():
        pred_new = model_new(ids, mask, c_type, c_place, c_feat, c_a, c_b).item()
    print(f"👉 예측값: {pred_new:.2f} %")
    
    lig_new = LayerIntegratedGradients(model_new, [
        model_new.bert.embeddings, model_new.emb_type, model_new.emb_place, model_new.emb_a, model_new.emb_b
    ])
    attr_new = lig_new.attribute(inputs=(ids, mask, c_type, c_place, c_feat, c_a, c_b), n_steps=50)
    
    score_new_text = attr_new[0].sum(dim=2).squeeze(0)
    score_new_text = score_new_text / torch.norm(score_new_text)
    
    # ---------------------------------------------------------
    # [결과 출력] Side-by-Side 비교
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'Token':<12} | {'Old Model Score':<15} | {'New Model Score':<15}")
    print("="*70)
    
    tokens = tok.convert_ids_to_tokens(ids[0])
    for t, s1, s2 in zip(tokens, score_old, score_new_text):
        if t == "[PAD]": break
        # 둘 중 하나라도 의미 있게 봤으면 출력 (절댓값 0.05 이상)
        if abs(s1.item()) > 0.05 or abs(s2.item()) > 0.05:
            print(f"{t:<12} | {s1.item():15.4f} | {s2.item():15.4f}")
            
    print("-" * 70)
    print("\n[New Model Special Features (Code Importance)]")
    print(f"Code A (직진)   Impact: {attr_new[3].sum().item():.4f}")
    print(f"Code B (좌회전) Impact: {attr_new[4].sum().item():.4f}")
    print(f"Code Type      Impact: {attr_new[1].sum().item():.4f}")

if __name__ == "__main__":
    main()