import json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import re

# ==========================================
# 1. 설정 (Config)
# ==========================================
MODEL_NAME = "bert-base-multilingual-cased"
LR, EPOCHS, BS, MAX_LEN = 2e-5, 50, 16, 256 # Epoch 50 유지
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 임베딩 크기 (넉넉하게)
NUM_TYPE = 1000
NUM_PLACE = 500
NUM_FEAT = 500
NUM_MOVE = 300
EMBED_DIM = 16    

# ==========================================
# 2. Hybrid Dataset (Strict Parsing 적용)
# ==========================================
class HybridParsingDataset(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tok
        
        # 🔴 [핵심 수정] 쉼표(,)를 넘어가지 못하게 `[^,]*?` 사용
        # 패턴 1: "(코드=123)" 형태
        # 패턴 2: "UNKNOWN(123)" 형태도 잡기 위해 `(?:코드=)?` 사용 (코드= 생략 가능)
        
        self.pat_type  = re.compile(r"사고유형=[^,]*?\(?(?:코드=)?(\d+)\)?")
        self.pat_place = re.compile(r"사고장소=[^,]*?\(?(?:코드=)?(\d+)\)?")
        self.pat_feat  = re.compile(r"장소특징=[^,]*?\(?(?:코드=)?(\d+)\)?")
        self.pat_a     = re.compile(r"A차량[^,]*?\(?(?:코드=)?(\d+)\)?")
        self.pat_b     = re.compile(r"B차량[^,]*?\(?(?:코드=)?(\d+)\)?")

    def extract_code(self, pattern, text, max_limit):
        m = pattern.search(text)
        if m:
            val = int(m.group(1))
            if val >= max_limit: return 0 # 범위 밖은 Unknown 처리
            return val
        return 0 # 없으면 0

    def __len__(self): return len(self.rows)
    
    def __getitem__(self, i):
        r = self.rows[i]
        text = r["text"]
        
        enc = self.tok(text, padding="max_length", truncation=True,
                       max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        
        # Strict Extraction
        item["c_type"]  = torch.tensor(self.extract_code(self.pat_type, text, NUM_TYPE), dtype=torch.long)
        item["c_place"] = torch.tensor(self.extract_code(self.pat_place, text, NUM_PLACE), dtype=torch.long)
        item["c_feat"]  = torch.tensor(self.extract_code(self.pat_feat, text, NUM_FEAT), dtype=torch.long)
        item["c_a"]     = torch.tensor(self.extract_code(self.pat_a, text, NUM_MOVE), dtype=torch.long)
        item["c_b"]     = torch.tensor(self.extract_code(self.pat_b, text, NUM_MOVE), dtype=torch.long)
        
        item["A"] = torch.tensor(float(r["A"]), dtype=torch.float)
        return item

# ==========================================
# 3. Hybrid Model (동일)
# ==========================================
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
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, c_type, c_place, c_feat, c_a, c_b):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]
        
        v_type = self.emb_type(c_type)
        v_place = self.emb_place(c_place)
        v_feat = self.emb_feat(c_feat)
        v_a = self.emb_a(c_a)
        v_b = self.emb_b(c_b)
        
        combined = torch.cat([h_cls, v_type, v_place, v_feat, v_a, v_b], dim=1)
        combined = self.dropout(combined)
        
        return self.regressor(combined).squeeze(-1)

# ==========================================
# 4. Main Loop (동일)
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
    ckpt_dir = Path("train/artifacts_hybrid")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    model = TrafficHybridRegressor(bert).to(DEVICE)

    ds_tr = HybridParsingDataset("train/train.jsonl", tok)
    ds_va = HybridParsingDataset("train/valid.jsonl", tok)
    
    dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BS)

    opt = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()
    
    print("=== Hybrid Training Start (Strict Parsing & Enhanced Regex) ===")
    
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
        print(f"Epoch {ep}/{EPOCHS} | Loss: {train_loss/len(dl_tr):.4f} | Valid MAE: {val_mae:.2f}")

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    tok.save_pretrained(str(ckpt_dir))
    bert.save_pretrained(str(ckpt_dir))
    print(f"Saved Hybrid Model to {ckpt_dir}")

if __name__ == "__main__":
    main()