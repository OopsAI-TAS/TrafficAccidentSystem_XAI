import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import re
from captum.attr import LayerIntegratedGradients

# ==========================================
# 1. 모델 정의 (Mask 확장 기능 포함)
# ==========================================
MODEL_NAME = "bert-base-multilingual-cased"
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
        # [Captum 배치처리 대응] Mask 자동 확장
        if input_ids is not None and attention_mask is not None:
            batch_size = input_ids.shape[0]
            if attention_mask.shape[0] != batch_size:
                attention_mask = attention_mask.expand(batch_size, -1)

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]
        
        v_type = self.emb_type(c_type); v_place = self.emb_place(c_place)
        v_feat = self.emb_feat(c_feat); v_a = self.emb_a(c_a); v_b = self.emb_b(c_b)
        
        combined = torch.cat([h_cls, v_type, v_place, v_feat, v_a, v_b], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        
        probs = F.softmax(logits, dim=-1)
        return probs[:, 0] * 100

# ==========================================
# 2. 실행 및 분석 로직
# ==========================================
def extract_code(pattern, text, max_limit):
    m = pattern.search(text)
    if m:
        val = int(m.group(1))
        return val if val < max_limit else 0
    return 0

def main():
    CKPT = Path("train/artifacts_hybrid_softmax")
    model_path = CKPT / "best_model.pt"
    
    if not model_path.exists():
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    bert = AutoModel.from_pretrained(str(CKPT))
    model = TrafficHybridSoftmax(bert)
    model.load_state_dict(torch.load(model_path))
    model.eval().cuda()

    # 테스트 데이터
    text = "[사고 정보] 사고유형=직진대좌회전(코드=214), 사고장소=교차로(코드=25) [차량 진행] A차량=직진(코드=6), B차량=좌회전(코드=14)"
    
    # Regex
    pat_type = re.compile(r"사고유형=[^,]*?\(?(?:코드=)?(\d+)\)?")
    pat_place = re.compile(r"사고장소=[^,]*?\(?(?:코드=)?(\d+)\)?")
    pat_feat  = re.compile(r"장소특징=[^,]*?\(?(?:코드=)?(\d+)\)?")
    pat_a     = re.compile(r"A차량[^,]*?\(?(?:코드=)?(\d+)\)?")
    pat_b     = re.compile(r"B차량[^,]*?\(?(?:코드=)?(\d+)\)?")

    # Input 준비
    enc = tok(text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
    input_ids = enc["input_ids"].cuda()
    mask = enc["attention_mask"].cuda() # 변수명: mask
    
    c_type = torch.tensor([extract_code(pat_type, text, NUM_TYPE)]).cuda()
    c_place = torch.tensor([extract_code(pat_place, text, NUM_PLACE)]).cuda()
    c_feat = torch.tensor([extract_code(pat_feat, text, NUM_FEAT)]).cuda()
    c_a = torch.tensor([extract_code(pat_a, text, NUM_MOVE)]).cuda()
    c_b = torch.tensor([extract_code(pat_b, text, NUM_MOVE)]).cuda()

    print(f"\n입력: {text}")
    print(f"파싱: Type={c_type.item()}, Place={c_place.item()}, A={c_a.item()}, B={c_b.item()}")

    # 예측
    with torch.no_grad():
        pred = model(input_ids, mask, c_type, c_place, c_feat, c_a, c_b)
    print(f"예측 과실비율(A): {pred.item():.2f}")

    # IG 분석
    print("\nCalculating Feature Importance (this may take a moment)...")
    lig = LayerIntegratedGradients(model, [
        model.bert.embeddings, 
        model.emb_type, model.emb_place, model.emb_a, model.emb_b
    ])
    
    # 🔴 [수정됨] inputs 튜플 안의 변수명을 'mask'로 변경
    attr = lig.attribute(
        inputs=(input_ids, mask, c_type, c_place, c_feat, c_a, c_b), 
        n_steps=50
    )
    
    # 결과 정리
    text_attr = attr[0].sum(dim=2).squeeze(0)
    text_attr = text_attr / torch.norm(text_attr)
    
    score_type = attr[1].sum().item()
    score_place = attr[2].sum().item()
    score_a = attr[3].sum().item()
    score_b = attr[4].sum().item()

    print("\n[📊 Feature Importance Comparison]")
    print("-" * 45)
    print(f"{'Source':<15} | {'Value':<10} | {'Importance':<10}")
    print("-" * 45)
    print(f"{'Code: A-Move':<15} | {c_a.item():<10} | {score_a:.4f}")
    print(f"{'Code: B-Move':<15} | {c_b.item():<10} | {score_b:.4f}")
    print(f"{'Code: Type':<15} | {c_type.item():<10} | {score_type:.4f}")
    print(f"{'Code: Place':<15} | {c_place.item():<10} | {score_place:.4f}")
    print("-" * 45)
    
    print("\n[📝 Top Text Tokens (BERT)]")
    tokens = tok.convert_ids_to_tokens(input_ids[0])
    print(f"{'Token':<15} | {'Score':<10}")
    print("-" * 30)
    
    for t, s in zip(tokens, text_attr):
        if t == "[PAD]": break
        if abs(s.item()) > 0.03: 
            print(f"{t:<15} | {s.item():.4f}")

if __name__ == "__main__":
    main()