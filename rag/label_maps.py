# # -*- coding: utf-8 -*-
# import json, pathlib
# from typing import Dict, Any

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# MAP_DIR = ROOT / "data" / "mappings"  # JSON 사전 위치

# # --- 안전한 빈 fallback (JSON이 없으면 빈 dict) ---
# EMPTY: Dict[int, str] = {}

# def _load_json_if_exists(path: pathlib.Path) -> Dict[int, str]:
#     if path.exists():
#         with open(path, "r", encoding="utf-8") as f:
#             raw = json.load(f)
#         return {int(k): v for k, v in raw.items()}
#     return {}

# def load_label_maps() -> Dict[str, Dict[int, str]]:
#     return {
#         "accident_object":        _load_json_if_exists(MAP_DIR / "accident_object.json")        or EMPTY,
#         "accident_place":         _load_json_if_exists(MAP_DIR / "accident_place.json")         or EMPTY,
#         "accident_place_feature": _load_json_if_exists(MAP_DIR / "accident_place_feature.json") or EMPTY,
#         "vehicle_a_progress_info":_load_json_if_exists(MAP_DIR / "vehicle_a_progress_info.json")or EMPTY,
#         "vehicle_b_progress_info":_load_json_if_exists(MAP_DIR / "vehicle_b_progress_info.json")or EMPTY,
#         "accident_type":          _load_json_if_exists(MAP_DIR / "accident_type.json")          or EMPTY,
#     }

# def label_value(maps: Dict[str, Dict[int, str]], key: str, code: Any) -> str:
#     try:
#         code_int = int(code)
#     except Exception:
#         return str(code)
#     d = maps.get(key, {})
#     return d.get(code_int, f"미정({code_int})")
# rag/label_maps.py
# -*- coding: utf-8 -*-
# import json
# import pathlib

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# MAPPINGS_DIR = ROOT / "data" / "mappings"

# def _load_json(file_name: str) -> dict:
#     """JSON 파일을 로드해서 dict 반환"""
#     path = MAPPINGS_DIR / file_name
#     if not path.exists():
#         raise FileNotFoundError(f"[ERR] Mapping file not found: {path}")
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # 각 매핑 로드
# ACCIDENT_OBJECT_MAP = _load_json("accident_object.json")
# ACCIDENT_PLACE_MAP = _load_json("accident_place.json")
# ACCIDENT_PLACE_FEATURE_MAP = _load_json("accident_place_feature.json")
# VEHICLE_A_PROGRESS_MAP = _load_json("accident_vehicle_a_progress_info.json")
# VEHICLE_B_PROGRESS_MAP = _load_json("accident_vehicle_b_progress_info.json")
# # 사고 유형은 별도 관리 (없으면 json 추가)
# ACCIDENT_TYPE_MAP = _load_json("accident_type.json") if (MAPPINGS_DIR / "accident_type.json").exists() else {}
# rag/label_maps.py
# -*- coding: utf-8 -*-
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAPPINGS_DIR = ROOT / "data" / "mappings"

def _load_json(file_name: str) -> dict:
    """JSON 파일을 로드해서 dict 반환 (키를 int로 변환)."""
    path = MAPPINGS_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"[ERR] Mapping file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 핵심: 키를 int로 변환
    return {int(k): v for k, v in raw.items()}

ACCIDENT_OBJECT_MAP = _load_json("accident_object.json")
ACCIDENT_PLACE_MAP = _load_json("accident_place.json")
ACCIDENT_PLACE_FEATURE_MAP = _load_json("accident_place_feature.json")
VEHICLE_A_PROGRESS_MAP = _load_json("accident_vehicle_a_progress_info.json")
VEHICLE_B_PROGRESS_MAP = _load_json("accident_vehicle_b_progress_info.json")
ACCIDENT_TYPE_MAP = _load_json("accident_type.json") if (MAPPINGS_DIR / "accident_type.json").exists() else {}