# train/train.py
import json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import os, shutil, tempfile, torch

MODEL_NAME = "bert-base-multilingual-cased"
LR, EPOCHS, BS, MAX_LEN = 2e-5, 3, 16, 256
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
        item["A"] = torch.tensor(r["A"], dtype=torch.float)
        return item

class TextOnlyHead(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.head = nn.Linear(bert.config.hidden_size, 2)  # logits(A,B)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]
        logits = self.head(h_cls)
        probs = torch.softmax(logits, dim=-1)
        predA = 100.0 * probs[:, 0]
        return predA

def evaluate(model, dl):
    model.eval(); mae=0; n=0
    with torch.no_grad():
        for b in dl:
            ids=b["input_ids"].to(DEVICE); mask=b["attention_mask"].to(DEVICE)
            y=b["A"].to(DEVICE)
            predA = model(ids, mask)
            mae += torch.abs(predA - y).sum().item(); n += y.size(0)
    return mae / max(n,1)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    model = TextOnlyHead(bert).to(DEVICE)

    ds_tr = JLDataset("train/train.jsonl", tok)
    ds_va = JLDataset("train/valid.jsonl", tok)
    dl_tr = DataLoader(ds_tr, batch_size=BS, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BS)

    opt = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    for ep in range(1, EPOCHS+1):
        model.train()
        for b in dl_tr:
            ids=b["input_ids"].to(DEVICE); mask=b["attention_mask"].to(DEVICE)
            y=b["A"].to(DEVICE)
            predA = model(ids, mask)
            loss = loss_fn(predA, y)
            opt.zero_grad(); loss.backward(); opt.step()
        val_mae = evaluate(model, dl_va)
        print(f"Epoch {ep}/{EPOCHS}  valid MAE(A) = {val_mae:.2f}")

    # 저장
    ckpt_dir = Path("train/artifacts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

# 1) 모델 헤드 가중치: 임시 파일에 저장 후 교체 + 레거시 포맷(버그 회피)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(tmp_fd)
    try:
        torch.save(model.state_dict(), tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, ckpt_dir / "model.pt")
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise e

    # 2) HF 토크나이저/베이스모델 저장 (이건 폴더 저장 방식)
    tok.save_pretrained(str(ckpt_dir))
    bert.save_pretrained(str(ckpt_dir))

    # 3) 저장 검증: 바로 로드해보기
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    print(f"[save-ok] artifacts at: {ckpt_dir.resolve()}")

if __name__ == "__main__":
    main()