import json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import re
import os

# ==========================================
# 1. 설정 (기존 분류 모델 파라미터 복원)
# ==========================================
MODEL_NAME = "bert-base-multilingual-cased"
LR, EPOCHS, BS, MAX_LEN = 2e-5, 30, 16, 256 # Epoch는 충분히 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 임베딩 설정
NUM_TYPE = 1000
NUM_PLACE = 500
NUM_FEAT = 500
NUM_MOVE = 300
EMBED_DIM = 32    

# ==========================================
# 2. Hybrid Dataset (파싱 로직 적용)
# ==========================================
class HybridParsingDataset(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tok
        
        # Strict Regex (탐욕 방지 적용됨)
        self.pat_type  = re.compile(r"사고유형=[^,]*?\(?(?:코드=)?(\d+)\)?")
        self.pat_place = re.compile(r"사고장소=[^,]*?\(?(?:코드=)?(\d+)\)?")
        self.pat_feat  = re.compile(r"장소특징=[^,]*?\(?(?:코드=)?(\d+)\)?")
        self.pat_a     = re.compile(r"A차량[^,]*?\(?(?:코드=)?(\d+)\)?")
        self.pat_b     = re.compile(r"B차량[^,]*?\(?(?:코드=)?(\d+)\)?")

    def extract_code(self, pattern, text, max_limit):
        m = pattern.search(text)
        if m:
            val = int(m.group(1))
            if val >= max_limit: return 0 
            return val
        return 0

    def __len__(self): return len(self.rows)
    
    def __getitem__(self, i):
        r = self.rows[i]
        text = r["text"]
        enc = self.tok(text, padding="max_length", truncation=True,
                       max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        
        # 코드 추출
        item["c_type"]  = torch.tensor(self.extract_code(self.pat_type, text, NUM_TYPE), dtype=torch.long)
        item["c_place"] = torch.tensor(self.extract_code(self.pat_place, text, NUM_PLACE), dtype=torch.long)
        item["c_feat"]  = torch.tensor(self.extract_code(self.pat_feat, text, NUM_FEAT), dtype=torch.long)
        item["c_a"]     = torch.tensor(self.extract_code(self.pat_a, text, NUM_MOVE), dtype=torch.long)
        item["c_b"]     = torch.tensor(self.extract_code(self.pat_b, text, NUM_MOVE), dtype=torch.long)
        
        item["A"] = torch.tensor(float(r["A"]), dtype=torch.float)
        return item

# ==========================================
# 3. Hybrid Softmax Model (기존 성공 모델의 확장판)
# ==========================================
class TrafficHybridSoftmax(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        
        # 임베딩 레이어
        self.emb_type = nn.Embedding(NUM_TYPE, EMBED_DIM)
        self.emb_place = nn.Embedding(NUM_PLACE, EMBED_DIM)
        self.emb_feat = nn.Embedding(NUM_FEAT, EMBED_DIM)
        self.emb_a = nn.Embedding(NUM_MOVE, EMBED_DIM)
        self.emb_b = nn.Embedding(NUM_MOVE, EMBED_DIM)
        
        # BERT(768) + Code(32*5 = 160) = 928
        combined_dim = bert.config.hidden_size + (EMBED_DIM * 5)
        
        # ★ 핵심 복원: 출력 2개 + Softmax 구조
        self.classifier = nn.Linear(combined_dim, 2) 

    def forward(self, input_ids, attention_mask, c_type, c_place, c_feat, c_a, c_b):
        # 1. Text
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]
        
        # 2. Code
        v_type = self.emb_type(c_type)
        v_place = self.emb_place(c_place)
        v_feat = self.emb_feat(c_feat)
        v_a = self.emb_a(c_a)
        v_b = self.emb_b(c_b)
        
        # 3. Concat
        combined = torch.cat([h_cls, v_type, v_place, v_feat, v_a, v_b], dim=1)
        combined = self.dropout(combined)
        
        # 4. Logits (A승리 확률, B승리 확률)
        logits = self.classifier(combined)
        
        # 5. Prediction Logic (기존 코드 로직 유지)
        probs = torch.softmax(logits, dim=-1)
        predA = 100.0 * probs[:, 0] # A의 과실비율 = A쪽 확률 * 100
        
        return predA

# ==========================================
# 4. Main Loop
# ==========================================
def evaluate(model, dl):
    model.eval()
    mae = 0; n = 0
    with torch.no_grad():
        for b in dl:
            args = [b[k].to(DEVICE) for k in ["input_ids", "attention_mask", "c_type", "c_place", "c_feat", "c_a", "c_b"]]
            y = b["A"].to(DEVICE)
            pred = model(*args)
            mae += torch.abs(pred - y).sum().item()
            n += y.size(0)
    return mae / max(n, 1)

def main():
    ckpt_dir = Path("train/artifacts_hybrid_softmax")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    
    # ★ 모델 교체
    model = TrafficHybridSoftmax(bert).to(DEVICE)

    ds_tr = HybridParsingDataset("train/train.jsonl", tok)
    ds_va = HybridParsingDataset("train/valid.jsonl", tok)
    dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BS)

    opt = optim.AdamW(model.parameters(), lr=LR)
    
    # Loss도 그대로 SmoothL1 유지 (Softmax 결과값 vs 실제값 비교)
    loss_fn = nn.SmoothL1Loss() 
    
    best_mae = float('inf')

    print("=== Hybrid Softmax Training Start (Returning to Legacy Structure) ===")
    
    for ep in range(1, EPOCHS+1):
        model.train()
        train_loss = 0
        for b in dl_tr:
            args = [b[k].to(DEVICE) for k in ["input_ids", "attention_mask", "c_type", "c_place", "c_feat", "c_a", "c_b"]]
            y = b["A"].to(DEVICE)
            
            pred = model(*args)
            loss = loss_fn(pred, y)
            
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
            
        val_mae = evaluate(model, dl_va)
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            save_msg = "✨ Best!"
        else:
            save_msg = ""

        print(f"Epoch {ep}/{EPOCHS} | Loss: {train_loss/len(dl_tr):.4f} | Valid MAE: {val_mae:.2f} {save_msg}")

    # 최종 저장
    tok.save_pretrained(str(ckpt_dir))
    bert.save_pretrained(str(ckpt_dir))
    print(f"Done! Best MAE: {best_mae:.2f}")

if __name__ == "__main__":
    main()