# # -*- coding: utf-8 -*-
# import json, argparse, pathlib
# from rag.query_builder import build_query
# from rag.retriever import load_retriever, search, format_context

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# PROMPT = (ROOT / "rag" / "prompts" / "legal_answer.txt").read_text(encoding="utf-8")

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--json", required=True, help="CV가 생성한 input JSON 경로")
#     ap.add_argument("--topk", type=int, default=8)
#     args = ap.parse_args()

#     inp = json.loads(pathlib.Path(args.json).read_text(encoding="utf-8"))
#     query = build_query(inp)
#     index, meta, model = load_retriever()
#     ctxs = search(index, meta, model, query, topk=args.topk)
#     context, a_ratio, b_ratio = format_context(ctxs)

#     prompt = PROMPT.format(question=query, context=context, a_ratio=a_ratio, b_ratio=b_ratio)
#     print("\n===== PROMPT =====\n")
#     print(prompt)

# if __name__ == "__main__":
#     main()

# rag/query.py
# -*- coding: utf-8 -*-
# import pathlib, sys, json
# from typing import List, Dict, Any

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# from rag.retriever import Retriever
# from rag.query_builder import load_input, build_query

# SAMPLE_INPUT = ROOT / "samples" / "input.json"
# PROMPT_TPL   = ROOT / "rag" / "prompts" / "legal_assistant.md"
# OUT_PROMPT   = ROOT / "data" / "index" / "request_prompt.txt"; OUT_PROMPT.parent.mkdir(parents=True, exist_ok=True)
# OUT_JSON     = ROOT / "data" / "index" / "retrieved.json"

# def _render_prompt(input_summary: str, snips: List[Dict[str, Any]]) -> str:
#     tpl = PROMPT_TPL.read_text(encoding="utf-8")
#     # 컨텍스트 슬림하게 정리
#     lines = []
#     for i, s in enumerate(snips, 1):
#         lines.append(f"[{i}] ({s.get('section')}, score={s.get('score'):.3f}) "
#                      f"ID={s.get('사고유형ID')} / {s.get('사고유형명')}\n{text_wrap(s.get('text',''))}")
#     ctx = "\n\n".join(lines)
#     return tpl.replace("{{INPUT_SUMMARY}}", input_summary)\
#               .replace("{{RETRIEVED_SNIPPETS}}", ctx)

# def text_wrap(t: str, width: int = 120) -> str:
#     out = []
#     line = ""
#     for ch in (t or ""):
#         line += ch
#         if len(line) >= width and ch == " ":
#             out.append(line); line = ""
#     if line: out.append(line)
#     return "\n".join(out)

# def main():
#     # 입력 불러오기
#     inp = load_input(SAMPLE_INPUT)
#     video = inp.get("video", {})
#     query, title = build_query(video)
#     video_name = video.get("video_name", "")
#     video_date = video.get("video_date", "")
#     input_summary = f"- video_name: {video_name}\n- video_date: {video_date}"

#     # 검색
#     ret = Retriever()
#     hits = ret.search(query, top_k=10)

#     # 산출물 저장
#     OUT_JSON.write_text(json.dumps({"query": query, "hits": hits}, ensure_ascii=False, indent=2), encoding="utf-8")

#     # 프롬프트 렌더링
#     prompt = _render_prompt(input_summary, hits)
#     OUT_PROMPT.write_text(prompt, encoding="utf-8")

#     print("="*60)
#     print("[QUERY]")
#     print(query)
#     print("="*60)
#     print("[INPUT SUMMARY]")
#     print(input_summary)
#     print("="*60)
#     print("[RETRIEVED TOP-K]")
#     for i, h in enumerate(hits, 1):
#         print(f"{i}. (score={h['score']:.3f}) "
#               f"{h.get('사고유형ID')} / {h.get('사고유형명')} / {h.get('section')}")
#         print("   ", h.get("text")[:180].replace("\n"," ") + "...")
#     print("="*60)
#     print(f"[PROMPT PREVIEW]\n{prompt[:500]}...\n")
#     print(f"[OK] query built → saved json={OUT_JSON}, prompt={OUT_PROMPT}")
#     print("="*60)

# if __name__ == "__main__":
#     main()
# rag/query_builder.py
# -*- coding: utf-8 -*-
import json, pathlib
from typing import Dict, Any, Tuple, List

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ★경로 통일: 모두 ROOT / "samples" / "input.json" 사용
SAMPLE_INPUT = ROOT / "samples" / "input.json"

OUT_DIR     = ROOT / "data" / "index"
OUT_QUERY   = OUT_DIR / "request_query.txt"
OUT_SUMMARY = OUT_DIR / "input_summary.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # None / 빈값이면 깔끔하게 "미정"만 표기
    if code in (None, "", []):
        return "미정"
    return f"{code}({label})"

def _show(x: Any) -> str:
    return "미정" if x in (None, "", []) else str(x)

def load_input(path: pathlib.Path = SAMPLE_INPUT) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["video"] if isinstance(data, dict) and "video" in data else data

def build_query(inp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
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

def main():
    inp = load_input(SAMPLE_INPUT)
    build_query(inp)

if __name__ == "__main__":
    main()

__all__ = ["load_input", "build_query"]