# rag/label_maps.py
import json, pathlib
from typing import Dict
from config.settings import get_progress_map_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "mappings"

def _load_json_soft(name: str) -> Dict[int, str]:
    path = BASE / name
    if not path.exists():
        print(f"[warn] mapping not found -> use empty: {path}")
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    for k, v in obj.items():
        try:
            k_int = int(k)
        except Exception:
            continue
        if isinstance(v, str):
            out[k_int] = v
        elif isinstance(v, dict):
            lab = v.get("label") or v.get("name") or str(v)
            out[k_int] = lab
        else:
            out[k_int] = str(v)
    return out

def _load_progress_code_to_label():
    mp = json.loads(get_progress_map_path().read_text(encoding="utf-8"))
    a_map, b_map = {}, {}
    for cat, codes in mp.get("vehicle_a_progress_info", {}).items():
        for c in codes or []:
            if c is None: continue
            try: a_map[int(c)] = cat
            except: pass
    for cat, codes in mp.get("vehicle_b_progress_info", {}).items():
        for c in codes or []:
            if c is None: continue
            try: b_map[int(c)] = cat
            except: pass
    return {"A": a_map, "B": b_map}

# ===== export maps =====
ACCIDENT_OBJECT_MAP        = _load_json_soft("accident_object.json")
ACCIDENT_PLACE_MAP         = _load_json_soft("accident_place.json")
ACCIDENT_PLACE_FEATURE_MAP = _load_json_soft("accident_place_feature.json")
ACCIDENT_TYPE_MAP          = _load_json_soft("accident_type.json")

_progress = _load_progress_code_to_label()
VEHICLE_A_PROGRESS_MAP = _progress["A"]
VEHICLE_B_PROGRESS_MAP = _progress["B"]