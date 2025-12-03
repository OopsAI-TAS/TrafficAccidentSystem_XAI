# train/prepare_dataset.py
import os, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
MAP_PATH = ROOT / "data" / "mappings" / "progress_maps.json"
OBJECT_MAP_PATH = "train/data/mappings/accident_object.json"
PLACE_FEATURE_MAP_PATH = "train/data/mappings/accident_place_feature.json"
PLACE_MAP_PATH = "train/data/mappings/accident_place.json"
A_PROGRESS_MAP_PATH = "train/data/mappings/accident_vehicle_a_progress_info.json"
B_PROGRESS_MAP_PATH = "train/data/mappings/accident_vehicle_b_progress_info.json"
OUT_TR = ROOT / "train.jsonl"
OUT_VA = ROOT / "valid.jsonl"

# ---------------------------------------------------------
# 1) 진행 정보(A/B진행) 매핑 로드 (기존 코드)
# ---------------------------------------------------------
def load_progress_maps():
    mp = json.load(open(MAP_PATH, "r", encoding="utf-8"))
    inv = {}

    for key in ["vehicle_a_progress_info", "vehicle_b_progress_info"]:
        inv[key] = {}
        for cat, codes in mp[key].items():
            for c in codes:
                inv[key][int(c)] = cat  # code -> category

    return inv


# ---------------------------------------------------------
# 2) 사고 유형/장소/장소특징 매핑 로드
# ---------------------------------------------------------
def load_basic_maps():
    type_map = json.load(open(OBJECT_MAP_PATH, "r", encoding="utf-8"))
    place_map = json.load(open(PLACE_MAP_PATH, "r", encoding="utf-8"))
    feat_map = json.load(open(PLACE_FEATURE_MAP_PATH, "r", encoding="utf-8"))
    return type_map, place_map, feat_map

# ---------------------------------------------------------
# 3) 자연어 텍스트 생성 부분
# ---------------------------------------------------------
def sample_to_text(s, progress_inv, type_map, place_map, feat_map):
    v = s.get("video", {})

    # 숫자 → 자연어 매핑
    type_raw = str(v.get("traffic_accident_type", ""))
    place_raw = str(v.get("accident_place", ""))
    feat_raw = str(v.get("accident_place_feature", ""))

    accident_type = type_map.get(type_raw, f"UNKNOWN({type_raw})")
    accident_place = place_map.get(place_raw, f"UNKNOWN({place_raw})")
    accident_place_feat = feat_map.get(feat_raw, f"UNKNOWN({feat_raw})")

    # A/B 진행 정보 (기존)
    a_code = v.get("vehicle_a_progress_info", None)
    b_code = v.get("vehicle_b_progress_info", None)
    a_cat = progress_inv["vehicle_a_progress_info"].get(a_code, "UNKNOWN")
    b_cat = progress_inv["vehicle_b_progress_info"].get(b_code, "UNKNOWN")

    # 최종 자연어 텍스트 구성
    txt = (
        f"[영상 정보] 촬영방식={v.get('filming_way','')}, 시점={v.get('video_point_of_view','')}, "
        f"영상명={v.get('video_name','')}, 촬영일자={v.get('video_date','')}\n"
        
        f"[사고 정보] 사고유형={accident_type}, 사고장소={accident_place}(코드={place_raw}), "
        f"장소특징={accident_place_feat}(코드={feat_raw})\n"
        
        f"[차량 진행] A차량진행상태={a_cat}(코드={a_code}), B차량진행상태={b_cat}(코드={b_code})"
    )

    return txt

# ---------------------------------------------------------
# 4) 메인: 데이터셋 생성
# ---------------------------------------------------------
def main():
    progress_inv = load_progress_maps()
    type_map, place_map, feat_map = load_basic_maps()

    rows = []

    for p in RAW_DIR.glob("*.json"):
        try:
            s = json.load(open(p, "r", encoding="utf-8"))
            txt = sample_to_text(s, progress_inv, type_map, place_map, feat_map)

            v = s.get("video", {})
            A = int(v.get("accident_negligence_rateA"))

            rows.append({"text": txt, "A": A})
        except Exception as e:
            print(f"[skip] {p.name}: {e}")

    # 셔플 + 분할
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

    print(f"[done] train={len(tr)} valid={len(va)} → {OUT_TR.name}, {OUT_VA.name}")

if __name__ == "__main__":
    main()