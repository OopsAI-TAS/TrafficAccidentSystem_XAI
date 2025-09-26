# 1번임 -> 제안과실비율이랑 기본과실표가 수정 덜 됨
# -*- coding: utf-8 -*-
# import os, sys, pathlib
# from dotenv import load_dotenv
# import json
# from typing import Dict, Any

# # .env 불러오기
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# IN_PROMPT  = ROOT / "data" / "index" / "request_prompt.txt"
# OUT_ANSWER = ROOT / "data" / "index" / "answer.txt"; OUT_ANSWER.parent.mkdir(parents=True, exist_ok=True)

# # 기본값: OpenAI 호환 엔드포인트 사용 (OpenAI, vLLM, OpenRouter 등)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# def call_openai(prompt: str) -> str:
#     """단일 프롬프트를 system 없이 user로 보내는 간단한 호출"""
#     import requests
#     url = f"{OPENAI_BASE_URL}/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {OPENAI_API_KEY}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "model": OPENAI_MODEL,
#         "messages": [
#             {"role":"system","content":"너는 한국어 법률 도우미다. 주어진 포맷을 정확히 따르고 근거를 명확히 요약해."},
#             {"role":"user","content": prompt}
#         ],
#         "temperature": 0.2,
#     }
#     r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
#     r.raise_for_status()
#     data = r.json()
#     return data["choices"][0]["message"]["content"].strip()

# # ... 기존 코드 상단 생략 ...

# def format_final_answer(result: Dict[str, Any], input_data: Dict[str, Any]) -> str:
#     video_name = input_data.get("video_name", "")
#     video_date = input_data.get("video_date", "")

#     lines = []
#     lines.append("## 요약 결론")
#     lines.append(f"- 제안 과실비율: A {result['A_ratio']}% / B {result['B_ratio']}%")
#     lines.append("")
#     lines.append("## 근거")
#     for g in result.get("grounds", []):
#         lines.append(f"- {g}")
#     lines.append("")
#     lines.append("## 적용 법조문")
#     for law in result.get("laws", []):
#         lines.append(f"- {law}")
#     lines.append("")
#     lines.append("## 참고 판례")
#     for pre in result.get("precedents", []):
#         lines.append(f"- {pre}")
#     lines.append("")
#     lines.append("## 입력 정보")
#     lines.append(f"- video_name: {video_name}")
#     lines.append(f"- video_date: {video_date}")

#     return "\n".join(lines)

# def main():
#     if not IN_PROMPT.exists():
#         print(f"[ERR] prompt not found: {IN_PROMPT}")
#         sys.exit(1)
#     prompt = IN_PROMPT.read_text(encoding="utf-8")

#     # API 키 없으면 프롬프트만 보여주고 종료 (개발용 프리뷰 모드)
#     if not OPENAI_API_KEY:
#         print("="*70)
#         print("[NO OPENAI_API_KEY] 프롬프트만 출력합니다. (모델 호출 없음)")
#         print("="*70)
#         print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
#         print("="*70)
#         print(f"(원하면 OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL 환경변수를 설정하세요)")
#         return

#     # 호출
#     print("[LLM] 호출 시작...")
#     ans = call_openai(prompt)
#     OUT_ANSWER.write_text(ans, encoding="utf-8")

#     # 터미널에도 찍기
#     print("="*70)
#     print("[FINAL ANSWER]")
#     print(ans)
#     print("="*70)
#     print(f"[OK] saved → {OUT_ANSWER}")

# if __name__ == "__main__":
#     main()

# rag/answer.py
# -*- coding: utf-8 -*-
#2번이고 기본 과실표가 없네  --> 계속 이거 썼음
# import os, sys, pathlib, re, json
# from typing import Dict, Any
# from dotenv import load_dotenv
# from rag.retriever import Retriever

# # .env 불러오기
# load_dotenv()

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# IN_PROMPT  = ROOT / "data" / "index" / "request_prompt.txt"
# IN_SUMMARY = ROOT / "data" / "index" / "input_summary.txt"
# OUT_ANSWER = ROOT / "data" / "index" / "answer.txt"; OUT_ANSWER.parent.mkdir(parents=True, exist_ok=True)

# # OpenAI 호환 설정
# OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  


# def build_context_with_required(query: str) -> str:
#     """
#     retriever.search()가 반환하는 meta 항목(섹션별 dict)을 모아
#     LLM이 반드시 참조해야 할 최소 컨텍스트(기본과실표/조정예시/법조/판례)를 구성한다.
#     """
#     retriever = Retriever()
#     results = retriever.search(query, top_k=16)

#     base_items = []     # [{"A":..,"B":..,"설명":...}]
#     adj_items  = []     # [{"대상":..,"가산사유":..,"조정값":..}]
#     law_items  = []     # [{"조문명":..,"핵심내용":..}]
#     pre_items  = []     # [{"출처":..,"판결요지":..}]

#     def _split_law(text: str):
#         # "도로교통법 제5조 – ...핵심내용" 형태 분리 (en dash/하이픈 모두 흡수)
#         parts = re.split(r"\s+[–-]\s+", text, maxsplit=1)
#         if len(parts) == 2:
#             return parts[0].strip(), parts[1].strip()
#         return text.strip(), ""

#     def _split_pre(text: str):
#         # "대법원 ... 판결 – ...요지" 형태 분리
#         parts = re.split(r"\s+[–-]\s+", text, maxsplit=1)
#         if len(parts) == 2:
#             return parts[0].strip(), parts[1].strip()
#         return text.strip(), ""

#     def _parse_adj_block(block_text: str):
#         # "- A차량 | A 현저한 과실 | +10%" 라인들을 dict로 파싱
#         out = []
#         for ln in block_text.splitlines():
#             ln = ln.strip()
#             if not ln or not ln.startswith("-"): 
#                 continue
#             # "- " 제거
#             ln = ln[1:].strip()
#             parts = [p.strip() for p in ln.split("|")]
#             if len(parts) >= 3:
#                 out.append({"대상": parts[0], "가산사유": parts[1], "조정값": parts[2]})
#         return out

#     # 섹션별로 모으기
#     for r in results:
#         sec  = r.get("section", "")
#         text = r.get("text", "") or ""
#         if not text:
#             continue

#         if sec == "기본과실비율_비율설명":
#             a = r.get("A_base")
#             b = r.get("B_base")
#             base_items.append({"A": a, "B": b, "설명": text})

#         elif sec == "과실비율조정예시":
#             adj_items.extend(_parse_adj_block(text))

#         elif sec == "적용법조항":
#             name, core = _split_law(text)
#             law_items.append({"조문명": name, "핵심내용": core})

#         elif sec == "참고판례":
#             src, gist = _split_pre(text)
#             pre_items.append({"출처": src, "판결요지": gist})

#     ctx_lines = []

#     # 기본과실표 (최상위 1건만 넣되, 설명 포함)
#     if base_items:
#         b = base_items[0]
#         ctx_lines.append("## 기본과실표")
#         if b.get("A") is not None and b.get("B") is not None:
#             ctx_lines.append(f"A={b['A']} / B={b['B']}")
#         if b.get("설명"):
#             ctx_lines.append(f"설명: {b['설명']}")

#     # 과실비율조정예시 (상위 4~8건 정도)
#     if adj_items:
#         ctx_lines.append("## 과실비율조정예시")
#         for x in adj_items[:8]:
#             ctx_lines.append(f"{x['대상']} | {x['가산사유']} | {x['조정값']}")

#     # 적용 법조문 (조문명 + 핵심내용)
#     if law_items:
#         ctx_lines.append("## 적용 법조문")
#         for l in law_items[:4]:
#             nm = l.get("조문명","")
#             core = l.get("핵심내용","")
#             ctx_lines.append(f"{nm} – {core}")

#     # 참고 판례 (출처 + 판결요지)
#     if pre_items:
#         ctx_lines.append("## 참고 판례")
#         for p in pre_items[:4]:
#             src = p.get("출처","")
#             gist = p.get("판결요지","")
#             ctx_lines.append(f"{src} – {gist}")

#     return "\n".join(ctx_lines)

# def call_openai(prompt: str) -> str:
#     """단일 프롬프트 호출"""
#     import requests
#     url = f"{OPENAI_BASE_URL}/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {OPENAI_API_KEY}",
#         "Content-Type": "application/json",
#     }
#     # 규칙 강화: A/B 합 100% 강제 및 포맷 고정
#     # sys_msg = (
#     #     "너는 한국어 법률 도우미다. 아래 포맷을 정확히 지키며, "
#     #     "‘제안 과실비율’은 반드시 A%+B%=100이 되도록 하라. "
#     #     "만약 내부 계산 결과의 합이 100이 아니면 적절히 정규화하여 100%가 되게 제시하고, "
#     #     "그 사실을 근거 섹션에 한 줄로 명시하라."
#     # )
#     # 
#     sys_msg = (
#     "너는 한국 교통사고 과실비율 도우미다. 반드시 아래 포맷을 따라라:\n"
#     "## 요약 결론\n"
#     "- 제안 과실비율: A xx% / B yy%\n\n"
#     "## 근거\n"
#     "- 기본과실표: 반드시 [검색 컨텍스트]의 '기본과실표'에서 가져온 설명 포함\n"
#     "- 조정예시 반영: 반드시 [검색 컨텍스트]의 '과실비율조정예시'에서 가져온 내용 포함\n\n"
#     "## 적용 법조문\n"
#     "- [검색 컨텍스트]의 '적용 법조문'에서 조문명 + 핵심내용 설명 포함\n\n"
#     "## 참고 판례\n"
#     "- [검색 컨텍스트]의 '참고 판례'에서 판례번호 + 판결요지 포함\n\n"
#     "## 입력 요약\n"
#     "- video_name: ...\n"
#     "- video_date: ..."
#     )
#     payload = {
#         "model": OPENAI_MODEL,
#         "messages": [
#             {"role":"system","content": sys_msg},
#             {"role":"user","content": prompt}
#         ],
#         "temperature": 0.2,
#     }
#     r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
#     r.raise_for_status()
#     data = r.json()
#     return data["choices"][0]["message"]["content"].strip()

# # ---------- 후처리: A/B 합 100% 정규화 & 근거 자동추가 ----------
# _AB_LINE_RE = re.compile(r"(제안\s*과실비율\s*:\s*A\s*)(\d{1,3})(\s*%\s*/\s*B\s*)(\d{1,3})(\s*%)")

# def _normalize_ratio_line(ans: str) -> str:
#     """
#     응답에서 '제안 과실비율: A xx% / B yy%'를 찾아,
#     합이 100이 아니면 비율 정규화 후 라인 교체 + 근거 섹션에 정규화 설명 한 줄 삽입.
#     """
#     m = _AB_LINE_RE.search(ans)
#     if not m:
#         return ans  # 포맷 라인이 없으면 스킵

#     a = int(m.group(2))
#     b = int(m.group(4))
#     total = a + b
#     if total == 100:
#         return ans

#     # 정규화 (가까운 정수, 합 100 보장)
#     new_a = round(a * 100.0 / total)
#     new_b = 100 - new_a

#     # 라인 치환
#     new_line = f"{m.group(1)}{new_a}{m.group(3)}{new_b}{m.group(5)}"
#     ans = ans[:m.start()] + new_line + ans[m.end():]

#     # 근거 섹션에 정규화 설명 삽입
#     norm_note = f"- 비율 정규화: 모델 산출 A {a}% / B {b}% (합 {total}%) → 총합 100%로 정규화하여 A {new_a}% / B {new_b}%로 조정"
#     ans = _inject_note_under_reason(ans, norm_note)
#     return ans

# def _inject_note_under_reason(ans: str, note_line: str) -> str:
#     """
#     '## 근거' 헤더 바로 아래에 note_line 한 줄 삽입.
#     헤더가 없으면 맨 끝에 섹션을 만들어 추가.
#     """
#     header = "## 근거"
#     pos = ans.find(header)
#     if pos == -1:
#         # 근거 섹션이 없으면 새로 만든다
#         if not ans.endswith("\n"):
#             ans += "\n"
#         ans += f"\n{header}\n{note_line}\n"
#         return ans

#     # 헤더 다음 줄 위치 찾기
#     line_end = ans.find("\n", pos + len(header))
#     if line_end == -1:
#         line_end = len(ans)
#     insert_at = line_end + 1
#     return ans[:insert_at] + note_line + "\n" + ans[insert_at:]

# # -------------------- 메인 --------------------
# def main():
#     if not IN_PROMPT.exists():
#         print(f"[ERR] prompt not found: {IN_PROMPT}")
#         sys.exit(1)
#     prompt_core = IN_PROMPT.read_text(encoding="utf-8")
#     prompt_sum  = IN_SUMMARY.read_text(encoding="utf-8") if IN_SUMMARY.exists() else ""

# # 필수 섹션 확보
#     context = build_context_with_required(prompt_core)

#     prompt = (
#         f"{prompt_core}\n\n"
#         f"[입력 요약]\n{prompt_sum}\n\n"
#         f"[검색 컨텍스트]\n{context}"
#     )

#     if not OPENAI_API_KEY:
#         print("="*70)
#         print("[NO OPENAI_API_KEY] 프롬프트만 출력합니다. (모델 호출 없음)")
#         print("="*70)
#         print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
#         print("="*70)
#         print(f"(OPENAI_API_KEY/OPENAI_MODEL/OPENAI_BASE_URL을 .env에 설정하세요)")
#         return

#     print("[LLM] 호출 시작...")
#     ans = call_openai(prompt)

#     # 안전장치: A/B 합 100%로 정규화 & 근거 삽입
#     ans = _normalize_ratio_line(ans)

#     OUT_ANSWER.write_text(ans, encoding="utf-8")

#     print("="*70)
#     print("[FINAL ANSWER]")
#     print(ans)
#     print("="*70)
#     print(f"[OK] saved → {OUT_ANSWER}")

# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
import os, sys, pathlib, re, json
from typing import Dict, Any
from dotenv import load_dotenv
from rag.retriever import Retriever

# .env 불러오기
load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

IN_PROMPT   = ROOT / "data" / "index" / "request_prompt.txt"
IN_SUMMARY  = ROOT / "data" / "index" / "input_summary.json"   # ← JSON으로 변경
OUT_ANSWER  = ROOT / "data" / "index" / "answer.txt"; OUT_ANSWER.parent.mkdir(parents=True, exist_ok=True)

# OpenAI 호환 설정
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def build_context_with_required(query: str) -> str:
    """
    벡터 검색 결과에서 기본과실표 / 조정예시 / 법조 / 판례를 모아
    LLM이 그대로 복사해 쓰기 쉽게 섹션으로 구성한다.
    상위 검색된 사고유형ID/이름 섹션도 함께 제공.
    """
    retriever = Retriever()
    results = retriever.search(query, top_k=32)
     #   === 디버그: 검색된 섹션 확인 ===
    # from collections import Counter
    # secs = [r.get("section", "") for r in results]
    # print("\n[DEBUG] 검색된 섹션 개수:", len(secs))
    # print("[DEBUG] 섹션 목록(앞 20개):", secs[:20])
    # print("[DEBUG] 섹션 빈도:", Counter(secs))
    # #각 결과 요약(앞 8개)
    # for i, r in enumerate(results[:8], 1):
    #     print(f"[DEBUG] #{i} section={r.get('section')}  caseID={r.get('사고유형ID')}  caseName={r.get('사고유형명')}")
    base_items, adj_items, law_items, pre_items = [], [], [], []

    # 상위 사고유형 ID/이름 수집
    top_cases = []  # [(id, name)]
    seen_ids = set()
    for r in results:
        cid = (r.get("사고유형ID") or "").strip()
        cname = (r.get("사고유형명") or "").strip()
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            top_cases.append((cid, cname))
        if len(top_cases) >= 8:
            break

    def _split_dash(text: str):
        # "이름 – 본문" / "이름 - 본문" 형태 분리
        parts = re.split(r"\s+[–-]\s+", text, maxsplit=1)
        return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (text.strip(), "")

    def _parse_adj_block(block_text: str):
        out = []
        for ln in block_text.splitlines():
            ln = ln.strip()
            if not ln.startswith("-"): 
                continue
            ln = ln[1:].strip()  # "- " 제거
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 3:
                out.append({"대상": parts[0], "가산사유": parts[1], "조정값": parts[2]})
        return out

    for r in results:
        sec  = (r.get("section", "") or "").strip()
        text = r.get("text", "") or ""
        if not text:
            continue

        # 기본과실표
        if sec.startswith("기본과실비율"):
            base_items.append({
                "A": r.get("A_base"),
                "B": r.get("B_base"),
                "설명": text
            })

        # 조정예시 (이름 변형까지 대응)
        elif any(key in sec for key in ["조정예시", "과실비율조정예시"]):
            adj_items.extend(_parse_adj_block(text))

        # 적용 법조문
        elif any(key in sec for key in ["적용법조항", "법규", "법조문"]):
            name, core = _split_dash(text)
            law_items.append({"조문명": name, "핵심내용": core})

        # 참고 판례
        elif any(key in sec for key in ["참고판례", "판례"]):
            src, gist = _split_dash(text)
            pre_items.append({"출처": src, "판결요지": gist})
    ctx = []

    if top_cases:
        ctx.append("## 검색된 사고유형(상위)")
        for cid, cname in top_cases:
            if cname:
                ctx.append(f"- {cid} – {cname}")
            else:
                ctx.append(f"- {cid}")

    if base_items:
        b = base_items[0]
        ctx.append("## 기본과실표")
        if b.get("A") is not None and b.get("B") is not None:
            ctx.append(f"A={b['A']} / B={b['B']}")
        if b.get("설명"):
            ctx.append(f"설명: {b['설명']}")

    if adj_items:
        ctx.append("## 과실비율조정예시")
        for x in adj_items[:8]:
            ctx.append(f"{x['대상']} | {x['가산사유']} | {x['조정값']}")

    if law_items:
        ctx.append("## 적용 법조문")
        for l in law_items[:4]:
            ctx.append(f"{l.get('조문명','')} – {l.get('핵심내용','')}")

    if pre_items:
        ctx.append("## 참고 판례")
        for p in pre_items[:4]:
            ctx.append(f"{p.get('출처','')} – {p.get('판결요지','')}")

    return "\n".join(ctx)

def call_openai(prompt: str) -> str:
    import requests
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    sys_msg = (
        "너는 한국 교통사고 과실비율 도우미다. 반드시 아래 포맷을 따라라:\n"
        "## 요약 결론\n"
        "- 제안 과실비율: A xx% / B yy%\n\n"
        "## 근거\n"
        "- 기본과실표: 반드시 [검색 컨텍스트]의 '기본과실표'에서 가져온 설명 포함\n"
        "- 조정예시 반영: 반드시 [검색 컨텍스트]의 '과실비율조정예시'에서 가져온 항목 최소 1개 이상 포함\n"
        "- (정규화 필요 시) 총합 100%로 정규화했다는 문장을 포함\n\n"
        "## 적용 법조문\n"
        "- [검색 컨텍스트]의 '적용 법조문'에서 조문명 + 핵심내용을 그대로 포함(의역 금지, 요약 가능)\n\n"
        "## 참고 판례\n"
        "- [검색 컨텍스트]의 '참고 판례'에서 출처 + 판결요지를 그대로 포함(의역 금지, 요약 가능)\n\n"
        "## 입력 요약\n"
        "- video_name: ...\n"
        "- video_date: ...\n"
        "과실비율은 반드시 A+B=100이 되도록 하라."
    )
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role":"system","content": sys_msg},
            {"role":"user","content": prompt}
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

# ---------- A/B 합 100% 정규화 ----------
_AB_LINE_RE = re.compile(r"(제안\s*과실비율\s*:\s*A\s*)(\d{1,3})(\s*%\s*/\s*B\s*)(\d{1,3})(\s*%)")

def _normalize_ratio_line(ans: str) -> str:
    m = _AB_LINE_RE.search(ans)
    if not m:
        return ans
    a = int(m.group(2)); b = int(m.group(4)); total = a + b
    if total == 100:
        return ans
    new_a = round(a * 100.0 / total)
    new_b = 100 - new_a
    new_line = f"{m.group(1)}{new_a}{m.group(3)}{new_b}{m.group(5)}"
    ans = ans[:m.start()] + new_line + ans[m.end():]
    note = f"- 비율 정규화: 모델 산출 A {a}% / B {b}% (합 {total}%) → 총합 100%로 정규화하여 A {new_a}% / B {new_b}%로 조정"
    ans = _inject_note_under_reason(ans, note)
    return ans

def _inject_note_under_reason(ans: str, note_line: str) -> str:
    header = "## 근거"
    pos = ans.find(header)
    if pos == -1:
        if not ans.endswith("\n"):
            ans += "\n"
        return ans + f"\n{header}\n{note_line}\n"
    line_end = ans.find("\n", pos + len(header))
    if line_end == -1:
        line_end = len(ans)
    insert_at = line_end + 1
    return ans[:insert_at] + note_line + "\n" + ans[insert_at:]

def _load_summary_fields() -> Dict[str, str]:
    """input_summary.json에서 video_name/date를 안전하게 읽어온다."""
    if IN_SUMMARY.exists():
        try:
            obj = json.loads(IN_SUMMARY.read_text(encoding="utf-8"))
            return {
                "video_name": obj.get("video_name", "미정"),
                "video_date": obj.get("video_date", "미정"),
            }
        except Exception:
            pass
    # 백업: 없거나 파싱 실패하면 미정
    return {"video_name": "미정", "video_date": "미정"}

# -------------------- 메인 --------------------
def main():
    if not IN_PROMPT.exists():
        print(f"[ERR] prompt not found: {IN_PROMPT}")
        sys.exit(1)

    prompt_core = IN_PROMPT.read_text(encoding="utf-8")

    # --- input_summary.json 직접 로드 ---
    video_name, video_date = "미정", "미정"
    if IN_SUMMARY.exists():
        try:
            obj = json.loads(IN_SUMMARY.read_text(encoding="utf-8"))
            video_name = obj.get("video_name", "미정")
            video_date = obj.get("video_date", "미정")
        except Exception as e:
            print("[WARN] input_summary.json 파싱 실패:", e)

    # 필수 섹션 확보
    context = build_context_with_required(prompt_core)

    # 최종 프롬프트
    prompt = (
        f"{prompt_core}\n\n"
        f"[입력 요약]\n"
        f"- video_name: {video_name}\n"
        f"- video_date: {video_date}\n\n"
        f"[검색 컨텍스트]\n{context}"
    )

    if not OPENAI_API_KEY:
        print("="*70)
        print("[NO OPENAI_API_KEY] 프롬프트만 출력합니다. (모델 호출 없음)")
        print("="*70)
        print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
        print("="*70)
        print("(OPENAI_API_KEY/OPENAI_MODEL/OPENAI_BASE_URL을 .env에 설정하세요)")
        return

    print("[LLM] 호출 시작...")
    ans = call_openai(prompt)

    # A/B 합 100% 정규화 & 근거 삽입
    ans = _normalize_ratio_line(ans)

    OUT_ANSWER.write_text(ans, encoding="utf-8")

    print("="*70)
    print("[FINAL ANSWER]")
    print(ans)
    print("="*70)
    print(f"[OK] saved → {OUT_ANSWER}")

if __name__ == "__main__":
    main()
