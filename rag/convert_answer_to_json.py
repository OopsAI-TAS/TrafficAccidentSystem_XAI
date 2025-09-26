# convert_answer_to_json.py
import json, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[0]
PROJ_ROOT = ROOT.parent   # rag/ 상위 = 프로젝트 루트
IN_FILE  = PROJ_ROOT / "data" / "index" / "answer.txt"
OUT_FILE = PROJ_ROOT / "data" / "index" / "answer.json"

def parse_answer(text: str):
    """answer.txt 내용을 JSON dict로 변환"""
    result = {
        "summary": {},
        "reasons": {"base_table": {}, "adjustments": []},
        "laws": [],
        "precedents": [],
        "input": {}
    }

    # --- 제안 과실비율 ---
    m = re.search(r"제안 과실비율:\s*A\s*(\d+)%\s*/\s*B\s*(\d+)%", text)
    if m:
        result["summary"] = {"A": int(m.group(1)), "B": int(m.group(2))}

    # --- 기본과실표 ---
    m = re.search(r"기본과실표:([^\n]+)", text)
    if m:
        result["reasons"]["base_table"]["desc"] = m.group(1).strip()

    # --- 조정예시 ---
    adj_block = re.search(r"조정예시 반영:(.*?)(?:\n\n|##|$)", text, re.S)
    if adj_block:
        for line in adj_block.group(1).splitlines():
            line = line.strip(" -")
            if not line: continue
            parts = re.split(r"[:：]", line, maxsplit=1)
            if len(parts) == 2:
                target_reason, delta = parts
                result["reasons"]["adjustments"].append({
                    "reason": target_reason.strip(),
                    "delta": delta.strip()
                })

    # --- 적용 법조문 ---
    law_block = re.search(r"## 적용 법조문(.*?)(?:##|$)", text, re.S)
    if law_block:
        for line in law_block.group(1).splitlines():
            line = line.strip(" -")
            if not line: continue
            parts = re.split(r"[–-]", line, maxsplit=1)
            if len(parts) == 2:
                result["laws"].append({
                    "name": parts[0].strip(),
                    "key": parts[1].strip()
                })

    # --- 참고 판례 ---
    pre_block = re.search(r"## 참고 판례(.*?)(?:##|$)", text, re.S)
    if pre_block:
        for line in pre_block.group(1).splitlines():
            line = line.strip(" -")
            if not line or "없" in line: continue
            parts = re.split(r"[–-]", line, maxsplit=1)
            if len(parts) == 2:
                result["precedents"].append({
                    "source": parts[0].strip(),
                    "gist": parts[1].strip()
                })

    # --- 입력 요약 ---
    m_name = re.search(r"video_name:\s*([^\n]+)", text)
    m_date = re.search(r"video_date:\s*([^\n]+)", text)
    result["input"] = {
        "video_name": m_name.group(1).strip() if m_name else "미정",
        "video_date": m_date.group(1).strip() if m_date else "미정"
    }

    return result

def main():
    if not IN_FILE.exists():
        print(f"[ERR] answer.txt not found at {IN_FILE}")
        return
    text = IN_FILE.read_text(encoding="utf-8")
    data = parse_answer(text)
    OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved JSON → {OUT_FILE}")

if __name__ == "__main__":
    main()