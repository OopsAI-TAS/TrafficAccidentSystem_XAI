import json, pathlib, os
from typing import Dict, Any, Tuple, List
from rag.retriever import Retriever
from config.settings import get_progress_map_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE_INPUT = ROOT / "rag" / "samples" / "input.json"

OUT_DIR     = ROOT / "data" / "index"
OUT_QUERY   = OUT_DIR / "request_query.txt"
OUT_SUMMARY = OUT_DIR / "input_summary.json"
OUT_RETRIEVED = OUT_DIR / "retrieved.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

section_bonus = {
    "조정예시_A측": 0.0,
    "조정예시_B측": 0.0,
}

section_bonus.update({
    "조정예시_공통": 0.0,
    "법조문": 0.0,
    "판례": 0.0,
})

from rag.label_maps import (
    ACCIDENT_OBJECT_MAP,
    ACCIDENT_PLACE_MAP,
    ACCIDENT_PLACE_FEATURE_MAP,
    VEHICLE_A_PROGRESS_MAP,
    VEHICLE_B_PROGRESS_MAP,
    ACCIDENT_TYPE_MAP,
)
def _load_progress_label_map():
    mp = json.loads(get_progress_map_path().read_text(encoding="utf-8"))
    invA, invB = {}, {}
    for cat, codes in mp["vehicle_a_progress_info"].items():
        for c in codes:
            if c is None: continue
            invA[int(c)] = cat
    for cat, codes in mp["vehicle_b_progress_info"].items():
        for c in codes:
            if c is None: continue
            invB[int(c)] = cat
    return invA, invB


def _lab(mapping: Dict[int, str], code: Any) -> str:
    try:
        return mapping.get(int(code), "미정")
    except Exception:
        return "미정"

def _pair(code: Any, label: str) -> str:
    # None / 빈값이면 깔끔하게 "미정"만 표기
    if code in (None, "", []):
        return "미정"
    return f"{code}({label})"

def _show(x: Any) -> str:
    return "미정" if x in (None, "", []) else str(x)

def load_input(path: pathlib.Path = SAMPLE_INPUT) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["video"] if isinstance(data, dict) and "video" in data else data

def build_query(inp):
    invA, invB = _load_progress_label_map()

    vid = _show(inp.get("video_name"))
    vdt = _show(inp.get("video_date"))

    a_pred = _show(inp.get("accident_negligence_rateA"))
    b_pred = _show(inp.get("accident_negligence_rateB"))

    obj_code   = inp.get("accident_object")
    place_code = inp.get("accident_place")
    feat_code  = inp.get("accident_place_feature")
    a_prog     = inp.get("vehicle_a_progress_info")
    b_prog     = inp.get("vehicle_b_progress_info")
    type_code  = inp.get("accident_type")

    a_lab = invA.get(int(a_prog), "미정") if a_prog not in (None,"",[]) else "미정"
    b_lab = invB.get(int(b_prog), "미정") if b_prog not in (None,"",[]) else "미정"

    obj_lab   = _lab(ACCIDENT_OBJECT_MAP,          obj_code)
    place_lab = _lab(ACCIDENT_PLACE_MAP,           place_code)
    feat_lab  = _lab(ACCIDENT_PLACE_FEATURE_MAP,   feat_code)
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

    # ===== 요약 JSON =====
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

    OUT_QUERY.write_text(query_text, encoding="utf-8")
    OUT_SUMMARY.write_text(json.dumps(input_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 프리뷰
    print("="*60)
    print("[QUERY (라벨 병기)]")
    print(query_text)
    print("="*60)
    print("[INPUT SUMMARY - JSON]")
    print(json.dumps(input_summary, ensure_ascii=False, indent=2))
    print("="*60)

    return query_text, input_summary

prior_path = ROOT / "data" / "index" / "prior.json"
if prior_path.exists():
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("A_ratio", 50) >= 60:
        section_bonus["조정예시_A측"] += 0.02
    elif prior.get("A_ratio", 50) <= 40:
        section_bonus["조정예시_B측"] += 0.02

def main():
    inp = load_input(SAMPLE_INPUT)
    query_text, input_summary = build_query(inp)

    ret = Retriever()
    results = ret.search(query_text, top_k=10, section_bonus=section_bonus)
    OUT_RETRIEVED.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("="*60)
    print("[RETRIEVED (with section_bonus)] top-10")
    print(json.dumps(results[:3], ensure_ascii=False, indent=2))  # 프리뷰 3개
    print("="*60)

if __name__ == "__main__":
    main()
