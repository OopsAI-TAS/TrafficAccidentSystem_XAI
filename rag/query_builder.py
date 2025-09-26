# # -*- coding: utf-8 -*-
# from typing import Dict, Any
# from ingestion.text_normalizer import normalize_ko

# OBJ = {0:"차량",1:"보행자",2:"이륜차/자전거"}
# POV = {0:"미상",1:"전방 블랙박스",2:"후방 블랙박스"}
# PLACE = {0:"교차로",1:"직선도로",2:"곡선도로"}
# # 필요시 계속 보완 (모르는 값은 코드 그대로 노출)

# def build_query(inp: Dict[str, Any]) -> str:
#     v = inp.get("video", {})
#     obj = OBJ.get(v.get("accident_object"), f"객체코드{v.get('accident_object')}")
#     place = PLACE.get(v.get("accident_place"), f"장소코드{v.get('accident_place')}")
#     pov = POV.get(v.get("video_point_of_view"), f"시점코드{v.get('video_point_of_view')}")
#     a = v.get("accident_negligence_rateA")
#     b = v.get("accident_negligence_rateB")
#     acc_type = v.get("accident_type")  # 코드 그대로도 힌트가 됨

#     # 한국어 자연어 질의 생성
#     q = f"""
#     블랙박스 시점:{pov}, 장소:{place}, 상대 객체:{obj}, 사고유형코드:{acc_type}.
#     CV가 추정한 과실비율 A:{a}% / B:{b}%.
#     유사한 사고유형의 기본 과실비율, 과실비율 조정예시, 적용 법조항, 참고 판례를 알려줘.
#     특히 과실비율 조정예시와 법조항·판례 요지를 요약해줘.
#     """
#     return normalize_ko(q.strip())

# rag/query_builder.py
# -*- coding: utf-8 -*-
## 2번째 -> rag/label_maps.py 넣기 전
# import json, pathlib
# from typing import Dict, Any, Tuple

# def load_input(input_path: pathlib.Path) -> Dict[str, Any]:
#     obj = json.loads(input_path.read_text(encoding="utf-8"))
#     return obj

# def build_query(video_obj: Dict[str, Any]) -> Tuple[str, str]:
#     """
#     video_obj: samples/input.json 의 "video" dict
#     반환: (free_text_query, compact_title)
#     """
#     v = video_obj
#     # 핵심 필드만 자연어로 묶기 (필요시 필드 추가/교체)
#     parts = []
#     def add(label, key):
#         if v.get(key) not in (None, "", []):
#             parts.append(f"{label}: {v.get(key)}")

#     add("사고유형코드", "accident_type")
#     add("A/B 과실(예측)", "accident_negligence_rateA")
#     add("A/B 과실(예측)", "accident_negligence_rateB")
#     add("장소", "accident_place")
#     add("장소특징", "accident_place_feature")
#     add("A 진행정보", "vehicle_a_progress_info")
#     add("B 진행정보", "vehicle_b_progress_info")
#     add("영상명", "video_name")
#     add("촬영일", "video_date")

#     query = " / ".join(parts)
#     title = f"{v.get('video_name','unknown')} ({v.get('video_date','')})"
#     return query, title


# rag/query_builder.py
# -*- coding: utf-8 -*-
# import json, pathlib
# from typing import Dict, Any, Tuple, List

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# SAMPLE_INPUT = ROOT / "samples" / "input.json"     # CV 쪽에서 주는 입력(샘플)
# OUT_QUERY    = ROOT / "data" / "index" / "query.txt"
# OUT_SUMMARY  = ROOT / "data" / "index" / "input_summary.txt"
# OUT_QUERY.parent.mkdir(parents=True, exist_ok=True)

# # 라벨 맵 가져오기 (완성본)
# from rag.label_maps import (
#     ACCIDENT_OBJECT_MAP,
#     ACCIDENT_PLACE_MAP,
#     ACCIDENT_PLACE_FEATURE_MAP,
#     VEHICLE_A_PROGRESS_MAP,
#     VEHICLE_B_PROGRESS_MAP,
#     ACCIDENT_TYPE_MAP,
# )

# def _lab(mapping: Dict[int, str], code: Any) -> str:
#     """정수 코드 → 라벨 문자열 (없으면 '미정')"""
#     try:
#         return mapping.get(int(code), "미정")
#     except Exception:
#         return "미정"

# def _pair(code: Any, label: str) -> str:
#     return f"{code}({label})"

# def build_query_text(inp: Dict[str, Any]) -> Tuple[str, str]:
#     """라벨 병기 질의문과, 짧은 입력 요약(사람 가독용)을 생성"""
#     vid = inp.get("video_name", "")
#     vdt = inp.get("video_date", "")

#     a_pred = inp.get("accident_negligence_rateA")
#     b_pred = inp.get("accident_negligence_rateB")

#     obj_code   = inp.get("accident_object")
#     place_code = inp.get("accident_place")
#     feat_code  = inp.get("accident_place_feature")
#     a_prog     = inp.get("vehicle_a_progress_info")
#     b_prog     = inp.get("vehicle_b_progress_info")
#     type_code  = inp.get("accident_type")

#     obj_lab   = _lab(ACCIDENT_OBJECT_MAP, obj_code)
#     place_lab = _lab(ACCIDENT_PLACE_MAP, place_code)
#     feat_lab  = _lab(ACCIDENT_PLACE_FEATURE_MAP, feat_code)
#     a_lab     = _lab(VEHICLE_A_PROGRESS_MAP, a_prog)
#     b_lab     = _lab(VEHICLE_B_PROGRESS_MAP, b_prog)
#     type_lab  = _lab(ACCIDENT_TYPE_MAP, type_code)

#     # ===== 질의용 텍스트(임베딩 대상) =====
#     lines: List[str] = []
#     lines.append("사고 질의(코드+라벨 보강)")
#     lines.append(f"- accident_type: { _pair(type_code,  type_lab) }")
#     lines.append(f"- accident_object: { _pair(obj_code,  obj_lab) }")
#     lines.append(f"- accident_place: { _pair(place_code, place_lab) }")
#     lines.append(f"- accident_place_feature: { _pair(feat_code,  feat_lab) }")
#     lines.append(f"- vehicle_a_progress_info: { _pair(a_prog, a_lab) }")
#     lines.append(f"- vehicle_b_progress_info: { _pair(b_prog, b_lab) }")
#     lines.append(f"- predicted_base_ratio: A={a_pred} / B={b_pred}")
#     # 비디오 메타는 검색어 가중엔 크게 도움은 적지만, 프롬프트 컨텍스트로 도움이 됨
#     lines.append(f"- video_name: {vid}")
#     lines.append(f"- video_date: {vdt}")
#     query_text = "\n".join(lines)

#     # ===== 터미널/프롬프트에 붙일 짧은 입력 요약(가독) =====
#     summary = (
#         f"video_name={vid}, video_date={vdt} | "
#         f"type={_pair(type_code, type_lab)}, object={_pair(obj_code, obj_lab)}, "
#         f"place={_pair(place_code, place_lab)}, feature={_pair(feat_code, feat_lab)} | "
#         f"A_prog={_pair(a_prog, a_lab)}, B_prog={_pair(b_prog, b_lab)} | "
#         f"pred A/B={a_pred}/{b_pred}"
#     )

#     return query_text, summary

# def main():
#     inp = json.loads(SAMPLE_INPUT.read_text(encoding="utf-8"))
#     q, s = build_query_text(inp["video"])
#     OUT_QUERY.write_text(q, encoding="utf-8")
#     OUT_SUMMARY.write_text(s, encoding="utf-8")
#     print("============================================================")
#     print("[QUERY (라벨 병기)]")
#     print(q)
#     print("============================================================")
#     print("[INPUT SUMMARY (라벨 병기)]")
#     print(s)
#     print("============================================================")
#     print(f"[OK] saved → {OUT_QUERY}, {OUT_SUMMARY}")

# if __name__ == "__main__":
#     main()

# rag/query_builder.py
# -*- coding: utf-8 -*-
# import json, pathlib
# from typing import Dict, Any, Tuple, List

# ROOT = pathlib.Path(__file__).resolve().parents[1]

# # 기본 샘플 입력 경로 (단독 실행 시 사용)
# SAMPLE_INPUT = ROOT / "rag" / "samples" / "input.json"   # ← samples 폴더는 rag 아래에 있음
# OUT_DIR     = ROOT / "data" / "index"
# OUT_QUERY   = OUT_DIR / "request_query.txt"              # 질의문 저장 위치 (query.py와 호환)
# OUT_SUMMARY = OUT_DIR / "input_summary.json"             # 입력 요약 저장 위치
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # 라벨 맵 (JSON에서 로드된 완성본 사전)
# from rag.label_maps import (
#     ACCIDENT_OBJECT_MAP,
#     ACCIDENT_PLACE_MAP,
#     ACCIDENT_PLACE_FEATURE_MAP,
#     VEHICLE_A_PROGRESS_MAP,
#     VEHICLE_B_PROGRESS_MAP,
#     ACCIDENT_TYPE_MAP,
# )

# def _lab(mapping: Dict[int, str], code: Any) -> str:
#     """정수 코드 → 라벨 문자열 (없으면 '미정')"""
#     try:
#         return mapping.get(int(code), "미정")
#     except Exception:
#         return "미정"

# def _pair(code: Any, label: str) -> str:
#     """코드(라벨) 표기"""
#     return f"{code}({label})"

# # ---------------------------------------------------------------------
# # 외부에서 import 하는 표준 API ① : 입력 로더
# # ---------------------------------------------------------------------
# def load_input(path: pathlib.Path = SAMPLE_INPUT) -> Dict[str, Any]:
#     """
#     CV 파트가 주는 입력 JSON을 로드해 video dict를 반환.
#     - { "video": {...} } 형태면 내부 "video"를 꺼내서 반환
#     - 이미 video dict면 그대로 반환
#     """
#     data = json.loads(path.read_text(encoding="utf-8"))
#     return data["video"] if isinstance(data, dict) and "video" in data else data

# # ---------------------------------------------------------------------
# # 외부에서 import 하는 표준 API ② : 질의문 생성기
# # ---------------------------------------------------------------------
# def build_query(inp: Dict[str, Any]) -> Tuple[str, str]:
#     """
#     라벨 병기 질의문과, 프롬프트에 붙일 짧은 입력 요약을 생성.
#     return: (query_text, input_summary)
#     """
#     vid = inp.get("video_name", "")
#     vdt = inp.get("video_date", "")

#     a_pred = inp.get("accident_negligence_rateA")
#     b_pred = inp.get("accident_negligence_rateB")

#     obj_code   = inp.get("accident_object")
#     place_code = inp.get("accident_place")
#     feat_code  = inp.get("accident_place_feature")
#     a_prog     = inp.get("vehicle_a_progress_info")
#     b_prog     = inp.get("vehicle_b_progress_info")
#     type_code  = inp.get("accident_type")

#     obj_lab   = _lab(ACCIDENT_OBJECT_MAP,          obj_code)
#     place_lab = _lab(ACCIDENT_PLACE_MAP,           place_code)
#     feat_lab  = _lab(ACCIDENT_PLACE_FEATURE_MAP,   feat_code)
#     a_lab     = _lab(VEHICLE_A_PROGRESS_MAP,       a_prog)
#     b_lab     = _lab(VEHICLE_B_PROGRESS_MAP,       b_prog)
#     type_lab  = _lab(ACCIDENT_TYPE_MAP,            type_code)

#     # ===== 질의용 텍스트(임베딩 대상) =====
#     lines: List[str] = []
#     lines.append("사고 질의(코드+라벨 보강)")
#     lines.append(f"- accident_type: {_pair(type_code,  type_lab)}")
#     lines.append(f"- accident_object: {_pair(obj_code,  obj_lab)}")
#     lines.append(f"- accident_place: {_pair(place_code, place_lab)}")
#     lines.append(f"- accident_place_feature: {_pair(feat_code,  feat_lab)}")
#     lines.append(f"- vehicle_a_progress_info: {_pair(a_prog, a_lab)}")
#     lines.append(f"- vehicle_b_progress_info: {_pair(b_prog, b_lab)}")
#     lines.append(f"- predicted_base_ratio: A={a_pred} / B={b_pred}")
#     # 비디오 메타는 검색어 가중에는 영향 적지만 컨텍스트에는 유용
#     lines.append(f"- video_name: {vid}")
#     lines.append(f"- video_date: {vdt}")
#     query_text = "\n".join(lines)

#     # ===== 요약 JSON (answer.py에서 읽을 수 있도록 구조화) =====
#     input_summary = {
#         "video_name": vid,
#         "video_date": vdt,
#         "type": _pair(type_code, type_lab),
#         "object": _pair(obj_code, obj_lab),
#         "place": _pair(place_code, place_lab),
#         "feature": _pair(feat_code, feat_lab),
#         "A_prog": _pair(a_prog, a_lab),
#         "B_prog": _pair(b_prog, b_lab),
#         "pred_AB": f"{a_pred}/{b_pred}"
#     }

#     # 파일로 저장(디버깅/검수용) - query.py와 경로 호환
#     OUT_QUERY.write_text(query_text, encoding="utf-8")
#     OUT_SUMMARY.write_text(input_summary, encoding="utf-8")

#     # 터미널 프리뷰
#     print("="*60)
#     print("[QUERY (라벨 병기)]")
#     print(query_text)
#     print("="*60)
#     print("[INPUT SUMMARY]")
#     print(input_summary)
#     print("="*60)

#     return query_text, input_summary

# # 단독 실행 지원 (샘플 입력으로 미리 만들어보기)
# def main():
#     inp = load_input(SAMPLE_INPUT)
#     build_query(inp)

# if __name__ == "__main__":
#     main()

# # 외부 모듈에서 import 할 공개 심볼
# __all__ = ["load_input", "build_query"]

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