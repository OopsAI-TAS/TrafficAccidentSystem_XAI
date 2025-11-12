# # import json
# # import pathlib

# # ROOT = pathlib.Path(__file__).resolve().parents[1]
# # MAPPINGS_DIR = ROOT / "data" / "mappings"

# # def _load_json(file_name: str) -> dict:
# #     """JSON 파일을 로드해서 dict 반환 (키를 int로 변환)."""
# #     path = MAPPINGS_DIR / file_name
# #     if not path.exists():
# #         raise FileNotFoundError(f"[ERR] Mapping file not found: {path}")
# #     with open(path, "r", encoding="utf-8") as f:
# #         raw = json.load(f)
# #     # 핵심: 키를 int로 변환
# #     return {int(k): v for k, v in raw.items()}

# # ACCIDENT_OBJECT_MAP = _load_json("accident_object.json")
# # ACCIDENT_PLACE_MAP = _load_json("accident_place.json")
# # ACCIDENT_PLACE_FEATURE_MAP = _load_json("accident_place_feature.json")
# # VEHICLE_A_PROGRESS_MAP = _load_json("accident_vehicle_a_progress_info.json")
# # VEHICLE_B_PROGRESS_MAP = _load_json("accident_vehicle_b_progress_info.json")
# # ACCIDENT_TYPE_MAP = _load_json("accident_type.json") if (MAPPINGS_DIR / "accident_type.json").exists() else {}
# # rag/label_maps.py
# import json
# import pathlib
# from typing import Dict, Any
# from config.settings import get_progress_map_path

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# BASE = ROOT / "data" / "mappings"

# def _load_json(name: str) -> Dict[int, str]:
#     path = BASE / name
#     if not path.exists():
#         raise FileNotFoundError(f"[ERR] Mapping file not found: {path}")
#     obj = json.loads(path.read_text(encoding="utf-8"))
#     # 파일 형식이 { "1":"라벨", "2":"라벨" } 또는 { "1": {... "label": "라벨"} } 등일 수 있으니
#     # 가장 단순한 {int: str} 형태로 정리
#     out = {}
#     for k, v in obj.items():
#         try:
#             k_int = int(k)
#         except Exception:
#             continue
#         if isinstance(v, str):
#             out[k_int] = v
#         elif isinstance(v, dict):
#             # v 안에 label 키가 있다면 사용
#             lab = v.get("label") or v.get("name") or str(v)
#             out[k_int] = lab
#         else:
#             out[k_int] = str(v)
#     return out

# def _load_progress_code_to_label() -> Dict[str, Dict[int, str]]:
#     """
#     progress_maps.json
#     {
#       "vehicle_a_progress_info": { "STRAIGHT": [2,3,...], ... },
#       "vehicle_b_progress_info": { "RIGHT_TURN": [ ... ], ... }
#     }
#     를 읽어 code->label(dict[int->str]) 로 변환
#     """
#     mp = json.loads(get_progress_map_path().read_text(encoding="utf-8"))
#     a_map: Dict[int, str] = {}
#     b_map: Dict[int, str] = {}

#     for cat, codes in mp.get("vehicle_a_progress_info", {}).items():
#         for c in codes:
#             if c is None: 
#                 continue
#             try:
#                 a_map[int(c)] = cat
#             except Exception:
#                 pass

#     for cat, codes in mp.get("vehicle_b_progress_info", {}).items():
#         for c in codes:
#             if c is None:
#                 continue
#             try:
#                 b_map[int(c)] = cat
#             except Exception:
#                 pass

#     return {"A": a_map, "B": b_map}

# # ====== 여기서부터 외부로 export 되는 맵들 ======
# ACCIDENT_OBJECT_MAP            = _load_json("accident_object.json")
# ACCIDENT_PLACE_MAP             = _load_json("accident_place.json")
# ACCIDENT_PLACE_FEATURE_MAP     = _load_json("accident_place_feature.json")
# ACCIDENT_TYPE_MAP              = _load_json("accident_type.json")

# # 진행방향은 progress_maps.json에서 코드→라벨로 만든다
# _progress = _load_progress_code_to_label()
# VEHICLE_A_PROGRESS_MAP = _progress["A"]
# VEHICLE_B_PROGRESS_MAP = _progress["B"]

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