# -*- coding: utf-8 -*-
import json, pathlib
from typing import Dict, Any, Tuple, List

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 기본 샘플 입력 경로
SAMPLE_INPUT = ROOT / "rag" / "samples" / "input.json"
OUT_DIR     = ROOT / "data" / "index"
OUT_QUERY   = OUT_DIR / "request_query.txt"
OUT_SUMMARY = OUT_DIR / "input_summary.json"   # ← JSON으로 저장하도록 변경
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 라벨 맵
from rag.label_maps import (
    ACCIDENT_OBJECT_MAP,
    ACCIDENT_PLACE_MAP,
    ACCIDENT_PLACE_FEATURE_MAP,
    VEHICLE_A_PROGRESS_MAP,
    VEHICLE_B_PROGRESS_MAP,
    ACCIDENT_TYPE_MAP,
)

def _lab(mapping: Dict[int, str], code: Any) -> str:
    try:
        return mapping.get(int(code), "미정")
    except Exception:
        return "미정"

def _pair(code: Any, label: str) -> str:
    return f"{code}({label})"

# ---------------------------------------------------------------------
def load_input(path: pathlib.Path = SAMPLE_INPUT) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["video"] if isinstance(data, dict) and "video" in data else data

# ---------------------------------------------------------------------
def build_query(inp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    vid = inp.get("video_name", "미정")
    vdt = inp.get("video_date", "미정")

    a_pred = inp.get("accident_negligence_rateA")
    b_pred = inp.get("accident_negligence_rateB")

    obj_code   = inp.get("accident_object")
    place_code = inp.get("accident_place")
    feat_code  = inp.get("accident_place_feature")
    a_prog     = inp.get("vehicle_a_progress_info")
    b_prog     = inp.get("vehicle_b_progress_info")
    type_code  = inp.get("accident_type")

    obj_lab   = _lab(ACCIDENT_OBJECT_MAP,          obj_code)
    place_lab = _lab(ACCIDENT_PLACE_MAP,           place_code)
    feat_lab  = _lab(ACCIDENT_PLACE_FEATURE_MAP,   feat_code)
    a_lab     = _lab(VEHICLE_A_PROGRESS_MAP,       a_prog)
    b_lab     = _lab(VEHICLE_B_PROGRESS_MAP,       b_prog)
    type_lab  = _lab(ACCIDENT_TYPE_MAP,            type_code)

    # ===== 질의문 =====
    lines: List[str] = []
    lines.append("사고 질의(코드+라벨 보강)")
    lines.append(f"- accident_type: {_pair(type_code,  type_lab)}")
    lines.append(f"- accident_object: {_pair(obj_code,  obj_lab)}")
    lines.append(f"- accident_place: {_pair(place_code, place_lab)}")
    lines.append(f"- accident_place_feature: {_pair(feat_code,  feat_lab)}")
    lines.append(f"- vehicle_a_progress_info: {_pair(a_prog, a_lab)}")
    lines.append(f"- vehicle_b_progress_info: {_pair(b_prog, b_lab)}")
    lines.append(f"- predicted_base_ratio: A={a_pred} / B={b_pred}")
    lines.append(f"- video_name: {vid}")
    lines.append(f"- video_date: {vdt}")
    query_text = "\n".join(lines)

    # ===== 요약 JSON (answer.py에서 읽을 수 있도록 구조화) =====
    input_summary = {
        "video_name": vid,
        "video_date": vdt,
        "type": _pair(type_code, type_lab),
        "object": _pair(obj_code, obj_lab),
        "place": _pair(place_code, place_lab),
        "feature": _pair(feat_code, feat_lab),
        "A_prog": _pair(a_prog, a_lab),
        "B_prog": _pair(b_prog, b_lab),
        "pred_AB": f"{a_pred}/{b_pred}"
    }

    # 저장
    OUT_QUERY.write_text(query_text, encoding="utf-8")
    OUT_SUMMARY.write_text(json.dumps(input_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 터미널 프리뷰
    print("="*60)
    print("[QUERY (라벨 병기)]")
    print(query_text)
    print("="*60)
    print("[INPUT SUMMARY - JSON]")
    print(json.dumps(input_summary, ensure_ascii=False, indent=2))
    print("="*60)

    return query_text, input_summary

# ---------------------------------------------------------------------
def main():
    inp = load_input(SAMPLE_INPUT)
    build_query(inp)

if __name__ == "__main__":
    main()

__all__ = ["load_input", "build_query"]