# train/infer.py
import json, torch
from pathlib import Path
from config.settings import get_progress_map_path
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
CKPT = Path(__file__).resolve().parent / "artifacts"

class TextOnlyHead(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.head = nn.Linear(bert.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:,0]
        logits = self.head(h_cls)
        probs = torch.softmax(logits, dim=-1)
        return 100.0 * probs[:,0]

def json_to_text(sample, inv):
    v = sample.get("video", {})
    a_code = v.get("vehicle_a_progress_info", None)
    b_code = v.get("vehicle_b_progress_info", None)
    a_cat = inv["vehicle_a_progress_info"].get(a_code, "UNKNOWN")
    b_cat = inv["vehicle_b_progress_info"].get(b_code, "UNKNOWN")
    return (
        f"[영상] 이름={v.get('video_name','')}, 날짜={v.get('video_date','')}, "
        f"촬영방식={v.get('filming_way','')}, 시점pov={v.get('video_point_of_view','')}\n"
        f"[사고] 유형코드={v.get('traffic_accident_type','')}, 장소={v.get('accident_place','')}, 장소특징={v.get('accident_place_feature','')}\n"
        f"[차량] A진행={a_cat}(코드={a_code}), B진행={b_cat}(코드={b_code})"
    )

def load_inv():
    mp_path = get_progress_map_path()
    mp = json.load(open(mp_path, "r", encoding="utf-8"))
    inv = {}
    for key in ["vehicle_a_progress_info","vehicle_b_progress_info"]:
        inv[key] = {}
        for cat, codes in mp[key].items():
            for c in codes:
                if c is None:  # 안전장치
                    continue
                inv[key][int(c)] = cat
    return inv

def predict(input_json_path, save_prior: bool = False):
    inv = load_inv()
    s = json.load(open(input_json_path, "r", encoding="utf-8"))
    text = json_to_text(s, inv)

    tok = AutoTokenizer.from_pretrained(str(CKPT))
    bert = AutoModel.from_pretrained(str(CKPT))
    model = TextOnlyHead(bert).to(DEVICE)
    model.load_state_dict(torch.load(CKPT / "model.pt", map_location=DEVICE))
    model.eval()

    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        predA = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
    A = int(round(predA.item()))
    B = 100 - A
    result = {"A_ratio": A, "B_ratio": B}

    if save_prior:
        save_to = ROOT / "data" / "index" / "prior.json"
        save_to.parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(save_to, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    return result

if __name__ == "__main__":
    # 예: python train/infer.py (파일 경로는 필요 시 하드코딩/수정)
    inp = ROOT / "rag" / "samples" / "input.json"
    print(predict(str(inp), save_prior=True))