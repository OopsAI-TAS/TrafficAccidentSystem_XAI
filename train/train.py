import json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import os, shutil, tempfile

# 설정
MODEL_NAME = "bert-base-multilingual-cased"
LR, EPOCHS, BS, MAX_LEN = 2e-5, 5, 16, 256 # Epoch를 3 -> 5로 조금 늘리는 것 추천
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class JLDataset(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.tok = tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], padding="max_length", truncation=True,
                       max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # Target shape: scalar (float)
        item["A"] = torch.tensor(r["A"], dtype=torch.float)
        return item

# 🔴 [핵심 수정] Softmax를 제거하고 순수 회귀(Regression) 모델로 변경
class TrafficRegressor(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        # 과적합 방지를 위한 Dropout 추가 (XAI 안정성 향상)
        self.dropout = nn.Dropout(0.1)
        # 출력 차원을 2 -> 1로 변경
        self.regressor = nn.Linear(bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS 토큰 추출
        h_cls = out.last_hidden_state[:, 0]
        h_cls = self.dropout(h_cls)
        
        # Softmax 없이 바로 실수값 예측 (0~100 사이의 값 학습)
        predA = self.regressor(h_cls)
        
        # 차원 축소: [Batch, 1] -> [Batch]
        return predA.squeeze(-1)

def evaluate(model, dl):
    model.eval()
    mae = 0
    n = 0
    with torch.no_grad():
        for b in dl:
            ids = b["input_ids"].to(DEVICE)
            mask = b["attention_mask"].to(DEVICE)
            y = b["A"].to(DEVICE)
            
            predA = model(ids, mask)
            
            # 예측값 범위 제한 (옵션: 0~100 벗어나면 잘라줌)
            # predA = torch.clamp(predA, 0, 100) 
            
            mae += torch.abs(predA - y).sum().item()
            n += y.size(0)
    return mae / max(n, 1)

def main():
    # 저장 경로 설정
    ckpt_dir = Path("train/artifacts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ❗ [주의] 모델 구조가 바뀌었으므로 기존 model.pt가 있다면 삭제하거나 무시해야 함
    # 아예 새로 학습하는 것이니 로드 과정 생략
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    
    # 모델 교체: TextOnlyHead -> TrafficRegressor
    model = TrafficRegressor(bert).to(DEVICE)

    ds_tr = JLDataset("train/train.jsonl", tok)
    ds_va = JLDataset("train/valid.jsonl", tok)
    dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BS)

    opt = optim.AdamW(model.parameters(), lr=LR)
    
    # Regression 손실함수
    loss_fn = nn.SmoothL1Loss() 

    print("=== Training Start (Regression Mode) ===")
    for ep in range(1, EPOCHS+1):
        model.train()
        train_loss = 0
        for b in dl_tr:
            ids = b["input_ids"].to(DEVICE)
            mask = b["attention_mask"].to(DEVICE)
            y = b["A"].to(DEVICE)
            
            predA = model(ids, mask)
            
            loss = loss_fn(predA, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            
        val_mae = evaluate(model, dl_va)
        print(f"Epoch {ep}/{EPOCHS} | Train Loss: {train_loss/len(dl_tr):.4f} | Valid MAE: {val_mae:.2f}")

    # 저장 (기존 로직 유지)
    print("Saving model...")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(tmp_fd)
    try:
        torch.save(model.state_dict(), tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, ckpt_dir / "model.pt")
    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise e

    tok.save_pretrained(str(ckpt_dir))
    bert.save_pretrained(str(ckpt_dir))
    
    print(f"[Done] Artifacts saved at: {ckpt_dir.resolve()}")

if __name__ == "__main__":
    main()