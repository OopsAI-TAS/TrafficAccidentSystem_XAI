# -*- coding: utf-8 -*-
import re, unicodedata

INVISIBLE_CHARS = (
    "\u200b\u200c\u200d\u200e\u200f"   # zero-width space/joiners/RTL marks
    "\u202a\u202b\u202c\u202d\u202e"   # embedding/override
    "\ufeff"                           # BOM
    "\u2060"                           # word joiner
    "\u00a0"                           # nbsp
    "\xad"                             # soft hyphen
)

INVISIBLE_RE = re.compile(f"[{re.escape(INVISIBLE_CHARS)}]")

MULTI_WS_RE = re.compile(r"[ \t\u00a0]+")
MULTI_NL_RE = re.compile(r"\n{3,}")

def normalize_ko(text: str) -> str:
    if not text:
        return text
    # 1) 유니코드 정규화
    t = unicodedata.normalize("NFKC", text)
    # 2) 보이지 않는 문자 제거
    t = INVISIBLE_RE.sub("", t)
    # 3) 하이픈 줄바꿈 잔재 정리
    t = t.replace("­", "")  # 일부 soft hyphen이 다른 코드로 남는 경우
    # 4) 공백 정리
    t = MULTI_WS_RE.sub(" ", t)
    t = MULTI_NL_RE.sub("\n\n", t)
    # 5) 좌우 공백
    return t.strip()