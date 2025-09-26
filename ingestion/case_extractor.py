import re, json, pathlib, fitz
from typing import List, Dict, Any, Optional

RAW_DIR = pathlib.Path("data/raw_pdfs")
OUT_DIR = pathlib.Path("data/law_json"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 공통 정규식 / 헤더
# ---------------------------
CASE_ID_PAT     = re.compile(r"\b(차|거)\d{1,3}-\d{1,3}\b")
CASE_TITLE_PAT  = re.compile(r"^\s*\d+\)\s*(.+사고.+)\s*$")
SCENARIO_A_PAT  = re.compile(r"^\(?A\)?\s*([^\n]+)$")
SCENARIO_B_PAT  = re.compile(r"^\(?B\)?\s*([^\n]+)$")

HDR_ACCIDENT    = re.compile(r"^\s*사고\s*상황\s*$")
HDR_BASIC_EXPL  = re.compile(r"^\s*기본\s*과실비율\s*(해설|해석)\s*$")
HDR_ADJ_EXPL    = re.compile(r"^\s*(수정요소|인과관계를\s*감안한\s*과실비율\s*조정|과실비율\s*조정\s*해설)\s*$")
HDR_LAW         = re.compile(r"^\s*(관련\s*법규|적용\s*법조항)\s*$")
HDR_PRECEDENT   = re.compile(r"^\s*참고\s*판례\s*$")
HDR_TABLE       = re.compile(r"^\s*기본\s*과실비율\s*$")

SECTION_HEADS   = [HDR_ACCIDENT, HDR_BASIC_EXPL, HDR_ADJ_EXPL, HDR_LAW, HDR_PRECEDENT, HDR_TABLE]
NOISE_STOP      = re.compile(r"^제\s*\d+\s*장|목차|자동차사고\s*과실비율\s*인정기준")

# A/B 기본 비율
AB_RATIO_INLINE = re.compile(r"A\s*([0-9]{1,3})\s*[^0-9A-Za-z가-힣]{0,8}\s*B\s*([0-9]{1,3})")

# 조정예시
ADJ_LINE_ALL    = re.compile(r"^\s*([AB])\s*([^\d+\-%]+?)\s*([+\-−＋]?\s*\d{1,2})\s*%?\s*$")
ADJ_ONLY_REASON = re.compile(r"^\s*([AB])\s*([^\d+\-%]+?)\s*$")
ADJ_ONLY_DELTA  = re.compile(r"^\s*([+\-−＋]?\s*\d{1,2})\s*%?\s*$")

# 법조항 이름
LAW_NAME_PAT    = re.compile(r"(도로교통법(?:\s*시행규칙)?(?:\s*별표\s*\d+)?\s*제?\s*\d*조?)")

# 판례 헤더(출처) 라인: '◎ ... 판결'
PRE_HEAD_PAT    = re.compile(r"^[◎◉○•\s]*([^\n]*?판결)\s*$")

# 원하는 요약 문구(정규화용)
LAW_SUMMARIES = {
    "도로교통법 제5조": "모든 운전자는 신호 또는 지시를 따라야 하며, 적색신호일 경우 정지해야 한다.",
    "도로교통법 제25조": "좌회전하려는 차량은 중앙선을 따라 서행하며 교차로 중심을 이용해야 하며, 신호가 있는 경우 그 지시에 따라야 한다.",
}

def _clean(s: str) -> str:
    s = s.replace("⊙"," ").replace("◎"," ◎ ").replace("◉"," ").replace("•"," ").replace("○"," ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def _normalize_title(title: str) -> str:
    title = re.sub(r"\s*\[[^\]]+\]\s*$", "", title)
    title = title.replace(" - ", " – ")
    return title.strip()

def _find_case_blocks(doc: fitz.Document) -> List[Dict[str, int]]:
    starts = []
    for pno in range(len(doc)):
        t = doc[pno].get_text("text")
        if CASE_ID_PAT.search(t) and ("기본 과실비율" in t or "(A" in t or "A)" in t):
            starts.append(pno)
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(doc)
        blocks.append({"p_start": s, "p_end": e})
    return blocks

def _slice_until(lines: List[str], start_idx: int) -> str:
    out = []
    for j in range(start_idx + 1, len(lines)):
        t = _clean(lines[j])
        if not t:
            continue
        if NOISE_STOP.search(t) or any(h.search(t) for h in SECTION_HEADS):
            break
        out.append(t)
        if len(out) > 60:
            break
    return " ".join(out)

def _extract_table_region(lines: List[str], start_idx: int) -> List[str]:
    region = []
    for j in range(start_idx + 1, len(lines)):
        t = _clean(lines[j])
        if NOISE_STOP.search(t) or any(h.search(t) for h in [HDR_ACCIDENT, HDR_BASIC_EXPL, HDR_ADJ_EXPL, HDR_LAW, HDR_PRECEDENT]):
            break
        if t:
            region.append(t)
        if len(region) > 160:
            break
    return region

def _parse_case_text(case_text: str) -> Optional[Dict[str, Any]]:
    lines = [ln for ln in case_text.splitlines()]
    joined = "\n".join(lines)

    # 1) 사고유형ID
    m_id = CASE_ID_PAT.search(joined)
    if not m_id:
        return None
    case_id = m_id.group()

    # 2) 제목
    title = None
    for ln in lines[:25]:
        mm = CASE_TITLE_PAT.search(ln)
        if mm:
            title = _normalize_title(_clean(mm.group(1)))
            break
    if not title:
        for ln in lines[:25]:
            if "사고" in ln:
                title = _normalize_title(_clean(ln)); break

    # 3) 시나리오 (A/B)
    a_desc, b_desc = None, None
    for ln in lines[:35]:
        t = _clean(ln)
        if a_desc is None:
            ma = SCENARIO_A_PAT.match(t)
            if ma: a_desc = ma.group(1)
        if b_desc is None:
            mb = SCENARIO_B_PAT.match(t)
            if mb: b_desc = mb.group(1)
        if a_desc and b_desc: break
    scenario = f"{a_desc or ''} A차량 vs {b_desc or ''} B차량".strip()

    # 4) 기본 과실비율 & 조정예시(표)
    base_ratio = {"A차량": None, "B차량": None}
    adjustments: List[Dict[str,str]] = []

    for i, ln in enumerate(lines):
        if HDR_TABLE.search(_clean(ln)):
            table_region = _extract_table_region(lines, i)
            region_text = "\n".join(table_region)

            # 기본 비율
            mratio = AB_RATIO_INLINE.search(region_text.replace(" ", ""))
            if mratio:
                base_ratio["A차량"] = int(mratio.group(1))
                base_ratio["B차량"] = int(mratio.group(2))

            # 조정예시: (1) 한 줄 형식
            for raw in table_region:
                t = _clean(raw).replace("＋", "+").replace("−", "-")
                m = ADJ_LINE_ALL.match(t)
                if m:
                    who_letter = m.group(1)
                    who = "A차량" if who_letter == "A" else "B차량"
                    reason = f"{who_letter} {m.group(2).strip(' .')}"
                    delta  = m.group(3).replace(" ", "")
                    if not delta.startswith(("+","-")): delta = "+" + delta
                    if not delta.endswith("%"): delta += "%"
                    adjustments.append({"대상": who, "가산사유": reason, "조정값": delta})

            # (2) 분리 열/줄 형식
            pending = None
            for raw in table_region:
                t = _clean(raw).replace("＋", "+").replace("−", "-")
                m_reason = ADJ_ONLY_REASON.match(t)
                m_delta  = ADJ_ONLY_DELTA.match(t)
                if m_reason:
                    pending = (m_reason.group(1), m_reason.group(2).strip(" ."))
                    continue
                if pending and m_delta:
                    who_letter, reason_text = pending
                    who = "A차량" if who_letter == "A" else "B차량"
                    delta = m_delta.group(1).replace(" ", "")
                    if not delta.startswith(("+","-")): delta = "+" + delta
                    if not delta.endswith("%"): delta += "%"
                    adjustments.append({"대상": who, "가산사유": f"{who_letter} {reason_text}", "조정값": delta})
                    pending = None
            break

    # 5) 본문 섹션
    accident_desc, basic_expl, adj_expl = "", "", ""
    for i, ln in enumerate(lines):
        t = _clean(ln)
        if HDR_ACCIDENT.search(t):
            accident_desc = _slice_until(lines, i)
        if HDR_BASIC_EXPL.search(t):
            basic_expl = _slice_until(lines, i)
        if HDR_ADJ_EXPL.search(t):
            adj_expl = _slice_until(lines, i)

    # 6) 관련 법규  ← 여기부터 교체
    laws: List[Dict[str, str]] = []
    in_law = False
    i = 0
    while i < len(lines):
        t = _clean(lines[i])

        # '관련 법규' 헤더 진입
        if not in_law and HDR_LAW.search(t):
            in_law = True
            i += 1
            continue

        if in_law:
            # 섹션 종료 조건
            if NOISE_STOP.search(t) or any(h.search(t) for h in [HDR_PRECEDENT, HDR_ACCIDENT, HDR_BASIC_EXPL]):
                break

            # 새로운 법조항 헤더(예: ◎ 도로교통법 제5조 …)
            m = LAW_NAME_PAT.search(t)
            if m:
                name = _clean(m.group(1))

                # 본문(다음 조문/다음 섹션 나오기 전까지) 수집
                body: List[str] = []
                j = i + 1
                while j < len(lines):
                    tt = _clean(lines[j])
                    if not tt:
                        j += 1
                        continue

                    # 다음 조문 헤더 또는 다른 섹션을 만나면 종료
                    if (LAW_NAME_PAT.search(tt) or
                        NOISE_STOP.search(tt) or
                        HDR_PRECEDENT.search(tt) or
                        HDR_ACCIDENT.search(tt) or
                        HDR_BASIC_EXPL.search(tt)):
                        break

                    body.append(tt)
                    j += 1

                laws.append({"조문명": name, "핵심내용": " ".join(body)})

                # 다음 루프로 계속(다음 조문 헤더를 다시 처리해야 하므로 i를 j로 이동)
                i = j
                continue

        i += 1

    # 7) 참고 판례 (헤더+요지 묶기) + 중복 제거
    precedents: List[Dict[str,str]] = []
    in_pre = False
    current_src = None
    current_body: List[str] = []

    def _norm_pair(src: str, body: str) -> str:
        return re.sub(r"\s+", " ", (src or "").strip()) + "||" + re.sub(r"\s+", " ", (body or "").strip())

    def _flush_pre():
        if current_src:
            body = " ".join(current_body).strip()
            body = re.sub(r"\s+", " ", body)
            pair_key = _norm_pair(current_src, body)
            if pair_key not in seen:
                precedents.append({"출처": current_src, "판결요지": body})
                seen.add(pair_key)

    seen = set()
    for i, ln in enumerate(lines):
        t = _clean(ln)
        if HDR_PRECEDENT.search(t):
            in_pre = True
            current_src = None
            current_body = []
            continue
        if in_pre:
            if NOISE_STOP.search(t) or any(h.search(t) for h in [HDR_LAW, HDR_ACCIDENT, HDR_BASIC_EXPL]):
                _flush_pre()
                break
            mh = PRE_HEAD_PAT.match(t)
            if mh:
                _flush_pre()  # 직전 건 확정
                current_src = mh.group(1)
                current_body = []
                continue
            if t:
                current_body.append(t)
    if in_pre:
        _flush_pre()

    # 8) 설명요약
    summary = ""
    if basic_expl:
        sents = re.split(r"(?<=[.!?。])\s+", basic_expl)
        summary = " ".join(sents[:2])[:500]
    elif accident_desc:
        summary = accident_desc[:240]

    return {
        "사고유형ID": case_id,
        "사고유형명": title or "",
        "사고도식": scenario,
        "사고상황설명": accident_desc,
        "기본과실비율": {
            "A차량": base_ratio["A차량"],
            "B차량": base_ratio["B차량"],
            "비율설명": basic_expl
        },
        "과실비율조정예시": adjustments,
        "적용법조항": laws,
        "참고판례": precedents,
        "설명요약": summary
    }

def _extract_text(doc: fitz.Document, p_start: int, p_end: int) -> str:
    buf = []
    for p in range(p_start, p_end):
        buf.append(doc[p].get_text("text"))
    return "\n".join(buf)

def _find_case_blocks(doc: fitz.Document) -> List[Dict[str, int]]:
    starts = []
    for pno in range(len(doc)):
        t = doc[pno].get_text("text")
        if CASE_ID_PAT.search(t) and ("기본 과실비율" in t or "(A" in t or "A)" in t):
            starts.append(pno)
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i+1] if i+1 < len(starts) else len(doc)
        blocks.append({"p_start": s, "p_end": e})
    return blocks

def extract_cases_from_pdf(pdf_path: pathlib.Path) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    cases = []
    for blk in _find_case_blocks(doc):
        text = _extract_text(doc, blk["p_start"], blk["p_end"])
        parsed = _parse_case_text(text)
        if parsed:
            cases.append(parsed)
    return cases

def main():
    out_path = OUT_DIR / "cases.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pdf in sorted(RAW_DIR.glob("*.pdf")):
            for c in extract_cases_from_pdf(pdf):
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
    print(f"[OK] extracted {n} cases → {out_path}")

if __name__ == "__main__":
    main()
