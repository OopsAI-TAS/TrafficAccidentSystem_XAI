import re, json, pathlib, fitz
from typing import List, Dict, Any, Optional

RAW_DIR = pathlib.Path("data/raw_pdfs")
OUT_DIR = pathlib.Path("data/law_json"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 패턴 정의 (케이스 전반)
# ---------------------------
CASE_ID_PAT      = re.compile(r"\b(차|거)\d{1,3}-\d{1,3}\b")   # 차3-1, 거2-1, 차55-2 ...
CASE_TITLE_PAT   = re.compile(r"^\s*\d+\)\s*(.+사고.+)\s*$")   # ex) 3) 직진 대 좌회전 사고 - ...
SUBTITLE_PAT     = re.compile(r"(직진|우회전|좌회전|횡단보도|녹색.*화살표|자전거|긴급자동차)")
SCENARIO_AB_PAT  = re.compile(r"\(\s*A\s*\)\s*([^\n]+)\s*\n\(\s*B\s*\)\s*([^\n]+)")  # (A) 직진 / (B) 좌회전
RATIO_CELL_PAT   = re.compile(r"\bA\s*([0-9]{1,3})\b.*?\bB\s*([0-9]{1,3})\b")       # A60 B40, A100 B0
ADJ_LINE_PAT     = re.compile(r"\b([AB])\s*(현저한 과실|중대한 과실|어린이·노인·장애인|서행불이행|야간·기타 시야장애|서행)\b.*?([+\-]\s*\d{1,2})")
LAW_HEAD_PAT     = re.compile(r"관련\s*법규|적용\s*법조항|도로교통법")
PRECEDENT_HEAD   = re.compile(r"참고\s*판례|판결")
ACCIDENT_HEAD    = re.compile(r"사고\s*상황")
BASIC_HEAD       = re.compile(r"기본\s*과실비율\s*해설|기본\s*과실비율\s*해석")
ADJ_HEAD         = re.compile(r"수정요소|인과관계.*과실비율\s*조정|과실비율\s*조정\s*예시")

def _clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def _find_case_blocks(doc: fitz.Document) -> List[Dict[str, Any]]:
    """페이지 단위 텍스트를 훑어 케이스 시작 지점을 찾고,
    다음 케이스가 나올 때까지를 하나의 block으로 묶는다."""
    starts = []
    for pno in range(len(doc)):
        txt = doc[pno].get_text("text")
        # 케이스 ID가 있고, 이미지/표와 함께 케이스 레이아웃일 가능성이 높은 페이지
        if CASE_ID_PAT.search(txt) and ( "기본 과실비율" in txt or "(A)" in txt ):
            starts.append(pno)

    blocks = []
    for i, p0 in enumerate(starts):
        p1 = starts[i+1] if i+1 < len(starts) else len(doc)  # 다음 케이스 시작 or 문서 끝
        blocks.append({"p_start": p0, "p_end": p1})
    return blocks

def _extract_text(doc: fitz.Document, p_start: int, p_end: int) -> str:
    buf = []
    for p in range(p_start, p_end):
        buf.append(doc[p].get_text("text"))
    return "\n".join(buf)

def _parse_case_text(case_text: str) -> Optional[Dict[str, Any]]:
    # 필수: 사고유형ID
    id_m = CASE_ID_PAT.search(case_text)
    if not id_m:
        return None
    case_id = id_m.group()

    # 상단 큰 제목(케이스명)
    title = None
    for ln in case_text.splitlines():
        if CASE_TITLE_PAT.search(ln):
            title = _clean(CASE_TITLE_PAT.search(ln).group(1))
            break
    if not title:
        # 제목 라인이 명확히 없을 때, 첫 줄들에서 유사 키워드
        for ln in case_text.splitlines()[:15]:
            if SUBTITLE_PAT.search(ln) and "사고" in ln:
                title = _clean(ln); break

    # (A)/(B) 시나리오 설명
    ab = SCENARIO_AB_PAT.search(case_text)
    scenario_text = None
    if ab:
        scenario_text = f"{_clean(ab.group(1))} A vs {_clean(ab.group(2))} B"

    # 기본 과실비율 Axx Bxx
    base_ratio = None
    mr = RATIO_CELL_PAT.search(case_text.replace(" ", ""))
    if mr:
        base_ratio = {"A차량": int(mr.group(1)), "B차량": int(mr.group(2))}

    # 조정 예시
    adjustments = []
    for m in ADJ_LINE_PAT.finditer(case_text.replace(" ", "")):
        who = "A차량" if m.group(1) == "A" else "B차량"
        reason = m.group(2)
        delta = m.group(3).replace(" ", "")
        # 퍼센트가 안 붙어있으면 % 붙여주기
        if not delta.endswith("%"): delta = delta + "%"
        adjustments.append({"대상": who, "가산사유": reason, "조정값": delta})

    # 사고상황/기본과실비율 해설/수정요소 설명
    accident_desc, basic_expl, adj_expl = None, None, None
    lines = case_text.splitlines()
    for i, ln in enumerate(lines):
        lns = _clean(ln)
        # 사고 상황
        if ACCIDENT_HEAD.search(lns):
            chunk = []
            for j in range(i+1, min(i+15, len(lines))):
                t = _clean(lines[j])
                if BASIC_HEAD.search(t) or ADJ_HEAD.search(t) or LAW_HEAD_PAT.search(t) or PRECEDENT_HEAD.search(t): break
                if t: chunk.append(t)
            if chunk: accident_desc = " ".join(chunk)
        # 기본과실비율 해설
        if BASIC_HEAD.search(lns):
            chunk = []
            for j in range(i+1, min(i+20, len(lines))):
                t = _clean(lines[j])
                if ACCIDENT_HEAD.search(t) or ADJ_HEAD.search(t) or LAW_HEAD_PAT.search(t) or PRECEDENT_HEAD.search(t): break
                if t: chunk.append(t)
            if chunk: basic_expl = " ".join(chunk)
        # 수정요소 해설
        if ADJ_HEAD.search(lns):
            chunk = []
            for j in range(i+1, min(i+20, len(lines))):
                t = _clean(lines[j])
                if ACCIDENT_HEAD.search(t) or BASIC_HEAD.search(t) or LAW_HEAD_PAT.search(t) or PRECEDENT_HEAD.search(t): break
                if t: chunk.append(t)
            if chunk: adj_expl = " ".join(chunk)

    # 법조항 및 판례 (현재 블록 및 다음 페이지 초반부까지 스캔했던 텍스트 안에서 키워드 매칭)
    laws, precedents = [], []
    # 간단 키워드 규칙: “도로교통법 제xx조”, “시행규칙 별표2”, “판결/판례/선고/대법원/고등법원”
    for ln in lines:
        if "도로교통법" in ln:
            m = re.search(r"(도로교통법(?: 시행규칙)?(?: 별표\d+)?\s*제?\d*조?)", ln)
            if m:
                laws.append({"조문명": _clean(m.group(1)), "핵심내용": _clean(ln)})
        if PRECEDENT_HEAD.search(ln) or "대법원" in ln or "고등법원" in ln or "판결" in ln or "선고" in ln:
            # 한 줄 요지 저장
            precedents.append({"출처": _clean(ln.split("판결")[0].replace("참고 판례", "").replace("참고", "")),
                               "판결요지": _clean(ln)})

    # 설명 요약 자동 초안
    summary = None
    if base_ratio and "적색" in case_text and "녹색" in case_text:
        # 전형적 직진-좌회전 신호 케이스 요약 템플릿
        if base_ratio["A차량"] == 100:
            summary = "A의 신호 위반이 주된 원인으로 A 일방 과실 사례. 다만 B의 주의의무 위반이 확인되면 일부 가산 가능."
    if not summary and basic_expl:
        summary = basic_expl[:200]

    # 최종 스키마 맵핑
    result = {
        "사고유형ID": case_id,
        "사고유형명": title or "",
        "사고도식": scenario_text or "",
        "사고상황설명": accident_desc or "",
        "기본과실비율": {
            "A차량": base_ratio["A차량"] if base_ratio else None,
            "B차량": base_ratio["B차량"] if base_ratio else None,
            "비율설명": basic_expl or ""
        },
        "과실비율조정예시": adjustments,
        "적용법조항": laws,
        "참고판례": precedents,
        "설명요약": summary or ""
    }
    return result

def extract_cases_from_pdf(pdf_path: pathlib.Path) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    cases = []
    for blk in _find_case_blocks(doc):
        txt = _extract_text(doc, blk["p_start"], blk["p_end"])
        parsed = _parse_case_text(txt)
        if parsed and parsed.get("사고유형ID"):
            cases.append(parsed)
    return cases

def main():
    out_path = OUT_DIR / "cases.jsonl"
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pdf in sorted(RAW_DIR.glob("*.pdf")):
            for c in extract_cases_from_pdf(pdf):
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                cnt += 1
    print(f"[OK] extracted {cnt} cases → {out_path}")

if __name__ == "__main__":
    main()