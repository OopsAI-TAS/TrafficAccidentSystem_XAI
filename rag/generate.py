# # # rag/generate.py
# # import json
# # import pathlib
# # import re
# # from collections import defaultdict
# # from typing import Dict, List, Tuple, Any

# # ROOT = pathlib.Path(__file__).resolve().parents[1]
# # OUT_DIR = ROOT / "data" / "index"
# # IN_SUMMARY = OUT_DIR / "input_summary.json"
# # IN_RETRIEVED = OUT_DIR / "retrieved.json"
# # OUT_DRAFT = OUT_DIR / "draft_opinion.txt"

# # # ===== 가감 규칙(간단 버전) =====
# # # 키워드 매칭으로 인정되는 가감치. 동일 근거 중복은 1회만 반영(cap_each=1).
# # ADJ_RULES: Dict[str, int] = {
# #     "현저한 과실": +5,
# #     "중대한 과실": +10,
# #     "명확한 선진입": -5,
# # }

# # # A/B 라인 식별
# # A_HEAD = re.compile(r"^\s*-\s*A차량\s*\|\s*A\s*")
# # B_HEAD = re.compile(r"^\s*-\s*B차량\s*\|\s*B\s*")

# # def _load_json(path: pathlib.Path, default: Any):
# #     try:
# #         return json.loads(path.read_text(encoding="utf-8"))
# #     except FileNotFoundError:
# #         print(f"[warn] file not found -> {path}")
# #         return default

# # def _parse_base_from_summary(summary: Dict[str, Any]) -> Tuple[int, int]:
# #     """input_summary.json의 pred_AB에서 기본 과실  (없으면 1순위 retrieved의 A_base/B_base, 최후엔 50/50)"""
# #     pred = summary.get("pred_AB")
# #     if isinstance(pred, str) and "/" in pred:
# #         try:
# #             a, b = pred.split("/", 1)
# #             a = int(str(a).strip())
# #             b = int(str(b).strip())
# #             if a + b == 100:
# #                 return a, b
# #         except Exception:
# #             pass
# #     return None, None  # 후순위로 미룸

# # def _parse_base_from_retrieved(docs: List[Dict[str, Any]]) -> Tuple[int, int]:
# #     for m in docs:
# #         a = m.get("A_base")
# #         b = m.get("B_base")
# #         if isinstance(a, int) and isinstance(b, int) and a + b == 100:
# #             return a, b
# #     return 50, 50

# # def parse_adjustments_from_docs(docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
# #     """
# #     retrieved.json의 text 라인에서 A/B별 키워드 카운트.
# #     동일 문서의 동일 키워드도 '횟수'는 증가시키지만, 실제 가감 계산 단계에서 cap_each로 1회만 인정.
# #     """
# #     counts = {"A": defaultdict(int), "B": defaultdict(int)}
# #     for m in docs:
# #         text = m.get("text") or ""
# #         for line in text.splitlines():
# #             who = None
# #             if A_HEAD.search(line):
# #                 who = "A"
# #             elif B_HEAD.search(line):
# #                 who = "B"
# #             if not who:
# #                 continue
# #             for key in ADJ_RULES.keys():
# #                 if key in line:
# #                     counts[who][key] += 1
# #     return counts

# # def compute_net_delta(counts: Dict[str, Dict[str, int]], cap_each: int = 1, max_abs_delta: int = 20) -> int:
# #     """
# #     A측 근거는 A에 불리(+), B측 근거는 A에 유리(-)로 정규화.
# #     cap_each=1 -> 동일 근거는 1회만 인정. 총합은 ±max_abs_delta% 캡, 5% 단위 반올림.
# #     """
# #     delta_A = 0
# #     delta_B = 0
# #     for k, v in counts["A"].items():
# #         delta_A += ADJ_RULES[k] * min(v, cap_each)
# #     for k, v in counts["B"].items():
# #         delta_B += ADJ_RULES[k] * min(v, cap_each)

# #     net = delta_A - delta_B  # (+)면 A% 증가, (-)면 A% 감소
# #     if net >  max_abs_delta: net =  max_abs_delta
# #     if net < -max_abs_delta: net = -max_abs_delta

# #     # 5% 단위 반올림
# #     net = int(round(net / 5.0)) * 5
# #     return net

# # def flatten_counts_for_report(counts: Dict[str, Dict[str, int]]) -> str:
# #     lines: List[str] = []
# #     for side in ("A", "B"):
# #         for k, v in counts[side].items():
# #             if v > 0:
# #                 lines.append(f"- {side}측 {k} (×{v})  -> 인정 {ADJ_RULES[k]}% × 1")
# #     return "\n".join(lines) if lines else "- (가감 인정 근거 없음)"

# # def build_case_header(summary: Dict[str, Any]) -> str:
# #     return (
# #         "[사건 개요]\n"
# #         f"- type: {summary.get('type','')}\n"
# #         f"- object: {summary.get('object','')}\n"
# #         f"- place: {summary.get('place','')}\n"
# #         f"- feature: {summary.get('feature','')}\n"
# #         f"- A_prog: {summary.get('A_prog','')}\n"
# #         f"- B_prog: {summary.get('B_prog','')}\n"
# #         f"- video_name: {summary.get('video_name','')}\n"
# #         f"- video_date: {summary.get('video_date','')}\n"
# #     )

# # def main():
# #     summary = _load_json(IN_SUMMARY, default={})
# #     docs = _load_json(IN_RETRIEVED, default=[])
# #     if not isinstance(docs, list):
# #         docs = []

# #     # 기본 과실 결정
# #     base_A, base_B = _parse_base_from_summary(summary)
# #     if base_A is None:
# #         base_A, base_B = _parse_base_from_retrieved(docs)

# #     # 근거 파싱 → 순가감치
# #     counts = parse_adjustments_from_docs(docs[:3])  # 상위 3개만 근거로 사용(너무 과도한 가감 방지)
# #     net = compute_net_delta(counts, cap_each=1, max_abs_delta=20)

# #     # 최종 권고
# #     A_rec = max(0, min(100, base_A + net))
# #     B_rec = 100 - A_rec

# #     # 리포트 구성
# #     header = build_case_header(summary)
# #     counts_block = flatten_counts_for_report(counts)

# #     report = f"""
# # {header}
# # [기본 과실비율]
# # A : {base_A}% / B : {base_B}%

# # [조정 근거(중복 병합)]
# # {counts_block}

# # [권고 과실비율]
# # 현 단계 권고 : A {A_rec}% / B {B_rec}%
# # (※ 동일 근거 중복은 1회만 인정, 총합 |Δ|≤20%로 제한, 5% 단위 반올림. 추가 사실관계 확인 시 변경 가능.)
# # """.strip()

# #     OUT_DRAFT.write_text(report, encoding="utf-8")

# #     # 콘솔 프리뷰
# #     print("==== [GENERATED DRAFT] ====")
# #     print(report)
# #     print(f"\n[save] {OUT_DRAFT}")

# # if __name__ == "__main__":
# #     main()

# # rag/generate.py
# import json
# import pathlib
# import re
# from collections import defaultdict, Counter
# from typing import Dict, List, Tuple, Any, Optional

# # -------------------- 경로 --------------------
# ROOT = pathlib.Path(__file__).resolve().parents[1]
# OUT_DIR = ROOT / "data" / "index"
# IN_SUMMARY = OUT_DIR / "input_summary.json"
# IN_RETRIEVED = OUT_DIR / "retrieved.json"
# OUT_DRAFT_TXT = OUT_DIR / "draft_opinion.txt"
# OUT_DRAFT_JSON = OUT_DIR / "draft_opinion.json"

# # 법/판례 원본(JSONL)
# CASES_PATH = ROOT / "data" / "law_json" / "cases.clean.jsonl"

# # -------------------- 가감 규칙(간단 버전) --------------------
# # 키워드 매칭으로 인정되는 가감치. 동일 근거 중복은 1회만 반영(cap_each=1).
# ADJ_RULES: Dict[str, int] = {
#     "현저한 과실": +5,
#     "중대한 과실": +10,
#     "명확한 선진입": -5,
# }

# # A/B 라인 식별용
# A_HEAD = re.compile(r"^\s*-\s*A차량\s*\|\s*A\s*")
# B_HEAD = re.compile(r"^\s*-\s*B차량\s*\|\s*B\s*")

# # -------------------- 유틸 --------------------
# def _load_json(path: pathlib.Path, default: Any):
#     try:
#         return json.loads(path.read_text(encoding="utf-8"))
#     except FileNotFoundError:
#         print(f"[warn] file not found -> {path}")
#         return default

# def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
#     items: List[Dict[str, Any]] = []
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     items.append(json.loads(line))
#                 except Exception:
#                     continue
#     except FileNotFoundError:
#         print(f"[warn] law jsonl not found -> {path}")
#     return items

# # -------------------- 기본 과실 파싱 --------------------
# def _parse_base_from_summary(summary: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
#     """input_summary.json의 pred_AB에서 기본 과실 (없으면 None)"""
#     pred = summary.get("pred_AB")
#     if isinstance(pred, str) and "/" in pred:
#         try:
#             a, b = pred.split("/", 1)
#             a = int(str(a).strip())
#             b = int(str(b).strip())
#             if a + b == 100:
#                 return a, b
#         except Exception:
#             pass
#     return None, None

# def _parse_base_from_retrieved(docs: List[Dict[str, Any]]) -> Tuple[int, int]:
#     """retrieved의 A_base/B_base 중 첫 유효 값, 없으면 50/50"""
#     for m in docs:
#         a = m.get("A_base")
#         b = m.get("B_base")
#         if isinstance(a, int) and isinstance(b, int) and a + b == 100:
#             return a, b
#     return 50, 50

# # -------------------- 가감 근거 파싱/계산 --------------------
# def parse_adjustments_from_docs(docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
#     """
#     retrieved.json의 text 라인에서 A/B별 키워드 카운트.
#     동일 문서의 동일 키워드는 '횟수' 카운트만 올리고, 실제 계산에서 cap_each로 1회만 인정.
#     """
#     counts = {"A": defaultdict(int), "B": defaultdict(int)}
#     for m in docs:
#         text = m.get("text") or ""
#         for line in text.splitlines():
#             who = "A" if A_HEAD.search(line) else ("B" if B_HEAD.search(line) else None)
#             if not who:
#                 continue
#             for key in ADJ_RULES.keys():
#                 if key in line:
#                     counts[who][key] += 1
#     return counts

# def compute_net_delta(counts: Dict[str, Dict[str, int]], cap_each: int = 1, max_abs_delta: int = 20) -> int:
#     """
#     A측 근거는 A에 불리(+), B측 근거는 A에 유리(-)로 정규화.
#     cap_each=1 -> 동일 근거는 1회만 인정. 총합은 ±max_abs_delta% 캡, 5% 단위 반올림.
#     """
#     delta_A = sum(ADJ_RULES[k] * min(v, cap_each) for k, v in counts["A"].items())
#     delta_B = sum(ADJ_RULES[k] * min(v, cap_each) for k, v in counts["B"].items())

#     net = delta_A - delta_B  # (+)면 A% 증가, (-)면 A% 감소
#     if net >  max_abs_delta: net =  max_abs_delta
#     if net < -max_abs_delta: net = -max_abs_delta

#     # 5% 단위 반올림
#     net = int(round(net / 5.0)) * 5
#     return net

# def flatten_counts_for_report(counts: Dict[str, Dict[str, int]]) -> str:
#     lines: List[str] = []
#     for side in ("A", "B"):
#         for k, v in counts[side].items():
#             if v > 0:
#                 lines.append(f"- {side}측 {k} (×{v})  -> 인정 {ADJ_RULES[k]}% × 1")
#     return "\n".join(lines) if lines else "- (가감 인정 근거 없음)"

# # -------------------- 리포트 헤더 --------------------
# def build_case_header(summary: Dict[str, Any]) -> str:
#     return (
#         "[사건 개요]\n"
#         f"- type: {summary.get('type','')}\n"
#         f"- object: {summary.get('object','')}\n"
#         f"- place: {summary.get('place','')}\n"
#         f"- feature: {summary.get('feature','')}\n"
#         f"- A_prog: {summary.get('A_prog','')}\n"
#         f"- B_prog: {summary.get('B_prog','')}\n"
#         f"- video_name: {summary.get('video_name','')}\n"
#         f"- video_date: {summary.get('video_date','')}\n"
#     )

# # -------------------- 법/판례 조인 --------------------
# def pick_majority_case_id(docs: List[Dict[str, Any]], top_k: int = 10) -> Optional[str]:
#     """retrieved 상위 문서들에서 가장 많이 등장한 사고유형ID 선택(동률이면 먼저 등장한 것 우선)"""
#     ids: List[str] = [d.get("사고유형ID") for d in docs[:top_k] if d.get("사고유형ID")]
#     if not ids:
#         return None
#     cnt = Counter(ids)
#     # 동률이면 리스트 상 먼저 나온 순서 우선
#     best_count = max(cnt.values())
#     for i in ids:
#         if cnt[i] == best_count:
#             return i
#     return None

# def load_cases_index() -> Dict[str, Dict[str, Any]]:
#     """사고유형ID -> 케이스 레코드 매핑"""
#     idx: Dict[str, Dict[str, Any]] = {}
#     for obj in _read_jsonl(CASES_PATH):
#         cid = obj.get("사고유형ID")
#         if cid:
#             idx[cid] = obj
#     return idx

# def extract_statutes(case_obj: Dict[str, Any]) -> List[Dict[str, str]]:
#     items = case_obj.get("적용법조항") or []
#     out = []
#     for it in items:
#         name = (it.get("조문명") or "").strip()
#         core = (it.get("핵심내용") or "").strip()
#         if name or core:
#             out.append({"조문명": name, "핵심내용": core})
#     return out

# def extract_precedents(case_obj: Dict[str, Any]) -> List[Dict[str, str]]:
#     items = case_obj.get("참고판례") or []
#     out = []
#     for it in items:
#         src = (it.get("출처") or "").strip()
#         gist = (it.get("판결요지") or "").strip()
#         if src or gist:
#             out.append({"출처": src, "판결요지": gist})
#     return out

# # -------------------- 메인 --------------------
# def main():
#     # 입력 로드
#     summary = _load_json(IN_SUMMARY, default={})
#     docs = _load_json(IN_RETRIEVED, default=[])
#     if not isinstance(docs, list):
#         docs = []

#     # 기본 과실 결정
#     base_A, base_B = _parse_base_from_summary(summary)
#     if base_A is None:
#         base_A, base_B = _parse_base_from_retrieved(docs)

#     # 근거 파싱 → 순가감치
#     counts = parse_adjustments_from_docs(docs[:3])  # 상위 3개만 근거로 사용(가감 과대 방지)
#     net = compute_net_delta(counts, cap_each=1, max_abs_delta=20)

#     # 최종 권고
#     A_rec = max(0, min(100, base_A + net))
#     B_rec = 100 - A_rec

#     # 리포트 기본 블록
#     header = build_case_header(summary)
#     counts_block = flatten_counts_for_report(counts)

#     lines: List[str] = []
#     lines.append(header)
#     lines.append("")
#     lines.append("[기본 과실비율]")
#     lines.append(f"A : {base_A}% / B : {base_B}%")
#     lines.append("")
#     lines.append("[조정 근거(중복 병합)]")
#     lines.append(counts_block)
#     lines.append("")
#     lines.append("[권고 과실비율]")
#     lines.append(f"현 단계 권고 : A {A_rec}% / B {B_rec}%")
#     lines.append("(※ 동일 근거 중복은 1회만 인정, 총합 |Δ|≤20%로 제한, 5% 단위 반올림. 추가 사실관계 확인 시 변경 가능.)")

#     # ----- 법/판례 추가 -----
#     used_case_id: Optional[str] = pick_majority_case_id(docs, top_k=10)
#     statutes: List[Dict[str, str]] = []
#     precedents: List[Dict[str, str]] = []

#     if used_case_id:
#         cases_idx = load_cases_index()
#         case_obj = cases_idx.get(used_case_id)
#         if case_obj:
#             statutes = extract_statutes(case_obj)
#             precedents = extract_precedents(case_obj)

#     lines.append("")
#     lines.append("[적용 법조항]")
#     if statutes:
#         for s in statutes[:5]:  # 너무 길면 상위 5개만
#             name = s.get("조문명", "").strip() or "(무명)"
#             gist = s.get("핵심내용", "").strip()
#             if gist:
#                 lines.append(f"- {name}: {gist}")
#             else:
#                 lines.append(f"- {name}")
#     else:
#         lines.append("- (관련 법조문 데이터 없음)")

#     lines.append("")
#     lines.append("[참고 판례]")
#     if precedents:
#         for p in precedents[:5]:
#             src = p.get("출처", "").strip() or "(출처 미상)"
#             gist = p.get("판결요지", "").strip()
#             if gist:
#                 lines.append(f"- {src}: {gist}")
#             else:
#                 lines.append(f"- {src}")
#     else:
#         lines.append("- (관련 판례 데이터 없음)")

#     # 텍스트 저장
#     report_txt = "\n".join(lines).strip()
#     OUT_DRAFT_TXT.write_text(report_txt, encoding="utf-8")

#     # JSON 저장(머신-리더블)
#     out_json = {
#         "case_summary": summary,
#         "base_ratio": {"A": base_A, "B": base_B},
#         "adjust_counts": {
#             "A": dict(counts["A"]),
#             "B": dict(counts["B"]),
#         },
#         "net_delta": net,
#         "recommendation": {"A": A_rec, "B": B_rec},
#         "used_case_id": used_case_id,
#         "statutes": statutes,
#         "precedents": precedents,
#         "retrieved_preview": docs[:3],  # 추적용 프리뷰
#     }
#     OUT_DRAFT_JSON.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

#     # 콘솔 프리뷰
#     print("==== [GENERATED DRAFT] ====")
#     print(report_txt)
#     print(f"\n[save] {OUT_DRAFT_TXT}")
#     print(f"[save] {OUT_DRAFT_JSON}")

# if __name__ == "__main__":
#     main()

# rag/generate.py
import json
import pathlib
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

# -------------------- 경로 --------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "index"
IN_SUMMARY = OUT_DIR / "input_summary.json"
IN_RETRIEVED = OUT_DIR / "retrieved.json"
OUT_DRAFT_TXT = OUT_DIR / "draft_opinion.txt"
OUT_DRAFT_JSON = OUT_DIR / "draft_opinion.json"

# 법/판례 원본(JSONL)
CASES_PATH = ROOT / "data" / "law_json" / "cases.clean.jsonl"

# -------------------- 가감 규칙(간단 버전) --------------------
# 키워드 매칭으로 인정되는 가감치. 동일 근거 중복은 1회만 반영(cap_each=1).
ADJ_RULES: Dict[str, int] = {
    "현저한 과실": +5,
    "중대한 과실": +10,
    "명확한 선진입": -5,
}

# A/B 라인 식별용
A_HEAD = re.compile(r"^\s*-\s*A차량\s*\|\s*A\s*")
B_HEAD = re.compile(r"^\s*-\s*B차량\s*\|\s*B\s*")

# -------------------- 유틸 --------------------
def _load_json(path: pathlib.Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[warn] file not found -> {path}")
        return default

def _read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        print(f"[warn] law jsonl not found -> {path}")
    return items

# -------------------- 기본 과실 파싱 --------------------
def _parse_base_from_summary(summary: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """input_summary.json의 pred_AB에서 기본 과실 (없으면 None)"""
    pred = summary.get("pred_AB")
    if isinstance(pred, str) and "/" in pred:
        try:
            a, b = pred.split("/", 1)
            a = int(str(a).strip())
            b = int(str(b).strip())
            if a + b == 100:
                return a, b
        except Exception:
            pass
    return None, None

def _parse_base_from_retrieved(docs: List[Dict[str, Any]]) -> Tuple[int, int]:
    """retrieved의 A_base/B_base 중 첫 유효 값, 없으면 50/50"""
    for m in docs:
        a = m.get("A_base")
        b = m.get("B_base")
        if isinstance(a, int) and isinstance(b, int) and a + b == 100:
            return a, b
    return 50, 50

# -------------------- 가감 근거 파싱/계산 --------------------
def parse_adjustments_from_docs(docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    retrieved.json의 text 라인에서 A/B별 키워드 카운트.
    동일 문서의 동일 키워드는 '횟수' 카운트만 올리고, 실제 계산에서 cap_each로 1회만 인정.
    """
    counts = {"A": defaultdict(int), "B": defaultdict(int)}
    for m in docs:
        text = m.get("text") or ""
        for line in text.splitlines():
            who = "A" if A_HEAD.search(line) else ("B" if B_HEAD.search(line) else None)
            if not who:
                continue
            for key in ADJ_RULES.keys():
                if key in line:
                    counts[who][key] += 1
    return counts

def compute_net_delta(counts: Dict[str, Dict[str, int]], cap_each: int = 1, max_abs_delta: int = 20) -> int:
    """
    A측 근거는 A에 불리(+), B측 근거는 A에 유리(-)로 정규화.
    cap_each=1 -> 동일 근거는 1회만 인정. 총합은 ±max_abs_delta% 캡, 5% 단위 반올림.
    """
    delta_A = sum(ADJ_RULES[k] * min(v, cap_each) for k, v in counts["A"].items())
    delta_B = sum(ADJ_RULES[k] * min(v, cap_each) for k, v in counts["B"].items())

    net = delta_A - delta_B  # (+)면 A% 증가, (-)면 A% 감소
    if net >  max_abs_delta: net =  max_abs_delta
    if net < -max_abs_delta: net = -max_abs_delta

    # 5% 단위 반올림
    net = int(round(net / 5.0)) * 5
    return net

def flatten_counts_for_report(counts: Dict[str, Dict[str, int]]) -> str:
    lines: List[str] = []
    for side in ("A", "B"):
        for k, v in counts[side].items():
            if v > 0:
                lines.append(f"- {side}측 {k} (×{v})  -> 인정 {ADJ_RULES[k]}% × 1")
    return "\n".join(lines) if lines else "- (가감 인정 근거 없음)"

# -------------------- 리포트 헤더 --------------------
def build_case_header(summary: Dict[str, Any]) -> str:
    return (
        "[사건 개요]\n"
        f"- type: {summary.get('type','')}\n"
        f"- object: {summary.get('object','')}\n"
        f"- place: {summary.get('place','')}\n"
        f"- feature: {summary.get('feature','')}\n"
        f"- A_prog: {summary.get('A_prog','')}\n"
        f"- B_prog: {summary.get('B_prog','')}\n"
        f"- video_name: {summary.get('video_name','')}\n"
        f"- video_date: {summary.get('video_date','')}\n"
    )

# -------------------- 법/판례 조인 (+키워드 백업) --------------------
def pick_majority_case_id(docs: List[Dict[str, Any]], top_k: int = 10) -> Optional[str]:
    """retrieved 상위 문서들에서 가장 많이 등장한 사고유형ID 선택(동률이면 먼저 등장한 것 우선)"""
    ids: List[str] = [d.get("사고유형ID") for d in docs[:top_k] if d.get("사고유형ID")]
    if not ids:
        return None
    cnt = Counter(ids)
    best_count = max(cnt.values())
    for i in ids:
        if cnt[i] == best_count:
            return i
    return None

def load_cases_index() -> Dict[str, Dict[str, Any]]:
    """사고유형ID -> 케이스 레코드 매핑"""
    idx: Dict[str, Dict[str, Any]] = {}
    for obj in _read_jsonl(CASES_PATH):
        cid = obj.get("사고유형ID")
        if cid:
            idx[cid] = obj
    return idx

def extract_statutes(case_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    items = case_obj.get("적용법조항") or []
    out = []
    for it in items:
        name = (it.get("조문명") or "").strip()
        core = (it.get("핵심내용") or "").strip()
        if name or core:
            out.append({"조문명": name, "핵심내용": core})
    return out

def extract_precedents(case_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    items = case_obj.get("참고판례") or []
    out = []
    for it in items:
        src = (it.get("출처") or "").strip()
        gist = (it.get("판결요지") or "").strip()
        if src or gist:
            out.append({"출처": src, "판결요지": gist})
    return out

def build_keywords(summary: Dict[str, Any], docs: List[Dict[str, Any]]) -> List[str]:
    """summary 라벨 + retrieved 텍스트에서 키워드 추출"""
    seeds: List[str] = []
    for k in ["type", "object", "feature", "A_prog", "B_prog"]:
        v = (summary.get(k) or "")
        # "97(차선변경)" -> 괄호 안 라벨만 추출
        if "(" in v and ")" in v:
            lab = v[v.find("(")+1:v.find(")")]
            seeds.append(lab)
        elif v:
            seeds.append(str(v))

    bag = Counter()
    for m in docs[:5]:
        txt = (m.get("text") or "") + " " + (m.get("사고유형명") or "") + " " + (m.get("사고상황설명") or "")
        for token in re.findall(r"[가-힣A-Za-z0-9]+", txt):
            if len(token) >= 2:
                bag[token] += 1

    prefer = ["자전거", "이륜차", "차선변경", "진로변경", "직진", "교차로"]
    kw = list(dict.fromkeys(prefer + seeds + [w for w, _ in bag.most_common(30)]))
    return kw

def search_cases_by_keywords(cases_idx: Dict[str, Dict[str, Any]], keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """간단한 키워드 매칭 점수로 연관 케이스 상위 top_k 반환"""
    scored = []
    for obj in cases_idx.values():
        text = " ".join([
            obj.get("사고유형ID",""),
            obj.get("사고유형명",""),
            obj.get("사고상황설명",""),
            json.dumps(obj.get("적용법조항") or "", ensure_ascii=False),
            json.dumps(obj.get("참고판례") or "", ensure_ascii=False),
        ])
        score = 0
        for kw in keywords:
            if kw and kw in text:
                score += 1
        if score > 0:
            scored.append((score, obj))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [o for _, o in scored[:top_k]]

def dedup_by_key(items: List[Dict[str, str]], key_fields: List[str]) -> List[Dict[str, str]]:
    """여러 케이스에서 수집된 법/판례 중복 제거"""
    seen = set()
    out = []
    for it in items:
        sig = tuple((it.get(k) or "").strip() for k in key_fields)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(it)
    return out

# -------------------- 메인 --------------------
def main():
    # 입력 로드
    summary = _load_json(IN_SUMMARY, default={})
    docs = _load_json(IN_RETRIEVED, default=[])
    if not isinstance(docs, list):
        docs = []

    # 기본 과실 결정
    base_A, base_B = _parse_base_from_summary(summary)
    if base_A is None:
        base_A, base_B = _parse_base_from_retrieved(docs)

    # 근거 파싱 → 순가감치
    counts = parse_adjustments_from_docs(docs[:3])  # 상위 3개만 근거로 사용(가감 과대 방지)
    net = compute_net_delta(counts, cap_each=1, max_abs_delta=20)

    # 최종 권고
    A_rec = max(0, min(100, base_A + net))
    B_rec = 100 - A_rec

    # 리포트 기본 블록
    header = build_case_header(summary)
    counts_block = flatten_counts_for_report(counts)

    lines: List[str] = []
    lines.append(header)
    lines.append("")
    lines.append("[기본 과실비율]")
    lines.append(f"A : {base_A}% / B : {base_B}%")
    lines.append("")
    lines.append("[조정 근거(중복 병합)]")
    lines.append(counts_block)
    lines.append("")
    lines.append("[권고 과실비율]")
    lines.append(f"현 단계 권고 : A {A_rec}% / B {B_rec}%")
    lines.append("(※ 동일 근거 중복은 1회만 인정, 총합 |Δ|≤20%로 제한, 5% 단위 반올림. 추가 사실관계 확인 시 변경 가능.)")

    # ----- 법/판례 추가 -----
    used_case_id: Optional[str] = pick_majority_case_id(docs, top_k=10)
    statutes: List[Dict[str, str]] = []
    precedents: List[Dict[str, str]] = []

    cases_idx = load_cases_index()

    # 1) 우선 retrieved 다수결 사고유형 사용
    if used_case_id and used_case_id in cases_idx:
        case_obj = cases_idx[used_case_id]
        statutes.extend(extract_statutes(case_obj))
        precedents.extend(extract_precedents(case_obj))

    # 2) 판례나 법조문이 비었으면 키워드 보강 검색으로 추가 확보
    if not statutes or not precedents:
        keywords = build_keywords(summary, docs)
        cand_cases = search_cases_by_keywords(cases_idx, keywords, top_k=5)
        for obj in cand_cases:
            statutes.extend(extract_statutes(obj))
            precedents.extend(extract_precedents(obj))

    # 3) 중복 제거 & 길이 제한
    statutes = dedup_by_key(statutes, ["조문명", "핵심내용"])[:5]
    precedents = dedup_by_key(precedents, ["출처", "판결요지"])[:5]

    lines.append("")
    lines.append("[적용 법조항]")
    if statutes:
        for s in statutes:
            name = s.get("조문명", "").strip() or "(무명)"
            gist = s.get("핵심내용", "").strip()
            if gist:
                lines.append(f"- {name}: {gist}")
            else:
                lines.append(f"- {name}")
    else:
        lines.append("- (관련 법조문 데이터 없음)")

    lines.append("")
    lines.append("[참고 판례]")
    if precedents:
        for p in precedents:
            src = p.get("출처", "").strip() or "(출처 미상)"
            gist = p.get("판결요지", "").strip()
            if gist:
                lines.append(f"- {src}: {gist}")
            else:
                lines.append(f"- {src}")
    else:
        lines.append("- (관련 판례 데이터 없음)")

    # 텍스트 저장
    report_txt = "\n".join(lines).strip()
    OUT_DRAFT_TXT.write_text(report_txt, encoding="utf-8")

    # JSON 저장(머신-리더블)
    out_json = {
        "case_summary": summary,
        "base_ratio": {"A": base_A, "B": base_B},
        "adjust_counts": {
            "A": dict(counts["A"]),
            "B": dict(counts["B"]),
        },
        "net_delta": net,
        "recommendation": {"A": A_rec, "B": B_rec},
        "used_case_id": used_case_id,
        "statutes": statutes,
        "precedents": precedents,
        "retrieved_preview": docs[:3],  # 추적용 프리뷰
    }
    OUT_DRAFT_JSON.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # 콘솔 프리뷰
    print("==== [GENERATED DRAFT] ====")
    print(report_txt)
    print(f"\n[save] {OUT_DRAFT_TXT}")
    print(f"[save] {OUT_DRAFT_JSON}")

if __name__ == "__main__":
    main()