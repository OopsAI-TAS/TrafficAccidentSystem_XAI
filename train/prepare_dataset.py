# train/prepare_dataset.py
import os, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
MAP_PATH = ROOT / "mappings" / "progress_maps.json"
OUT_TR = ROOT / "train.jsonl"
OUT_VA = ROOT / "valid.jsonl"

def load_maps():
    mp = json.load(open(MAP_PATH, "r", encoding="utf-8"))
    inv = {}
    for key in ["vehicle_a_progress_info", "vehicle_b_progress_info"]:
        inv[key] = {}
        for cat, codes in mp[key].items():
            for c in codes:
                inv[key][int(c)] = cat  # code -> category
    return inv

def sample_to_text(s, inv):
    v = s.get("video", {})
    a_code = v.get("vehicle_a_progress_info", None)
    b_code = v.get("vehicle_b_progress_info", None)
    a_cat = inv["vehicle_a_progress_info"].get(a_code, "UNKNOWN")
    b_cat = inv["vehicle_b_progress_info"].get(b_code, "UNKNOWN")
    # 간단한 한국어 템플릿 (BERT용)
    txt = (
        f"[영상] 이름={v.get('video_name','')}, 날짜={v.get('video_date','')}, "
        f"촬영방식={v.get('filming_way','')}, 시점pov={v.get('video_point_of_view','')}\n"
        f"[사고] 유형코드={v.get('traffic_accident_type','')}, 장소={v.get('accident_place','')}, 장소특징={v.get('accident_place_feature','')}\n"
        f"[차량] A진행={a_cat}(코드={a_code}), B진행={b_cat}(코드={b_code})"
    )
    return txt

def main():
    inv = load_maps()
    rows = []
    for p in RAW_DIR.glob("*.json"):
        try:
            s = json.load(open(p, "r", encoding="utf-8"))
            txt = sample_to_text(s, inv)
            v = s.get("video", {})
            A = int(v.get("accident_negligence_rateA"))
            # JSONL 한 줄
            rows.append({"text": txt, "A": A})
        except Exception as e:
            print(f"[skip] {p.name}: {e}")

    # 셔플 + 분할(8:2)
    random.shuffle(rows)
    n = len(rows)
    k = max(1, int(n * 0.8))
    tr, va = rows[:k], rows[k:]

    with open(OUT_TR, "w", encoding="utf-8") as f:
        for r in tr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_VA, "w", encoding="utf-8") as f:
        for r in va:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] train={len(tr)} valid={len(va)} -> {OUT_TR.name}, {OUT_VA.name}")

if __name__ == "__main__":
    main()