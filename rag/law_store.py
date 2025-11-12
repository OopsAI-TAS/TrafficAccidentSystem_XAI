# rag/law_store.py
import json, pathlib
from typing import Dict, Any, List, Optional
ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "law_json" / "cases.clean.jsonl"

def _load_cases() -> Dict[str, Dict[str, Any]]:
    """사고유형ID -> 전체 레코드 매핑으로 로드"""
    cases: Dict[str, Dict[str, Any]] = {}
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("사고유형ID")
                if cid:
                    cases[cid] = obj
            except Exception:
                continue
    return cases

# 프로세스 내 1회 로드 후 캐시
_CASES_CACHE: Optional[Dict[str, Dict[str, Any]]] = None

def get_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    global _CASES_CACHE
    if _CASES_CACHE is None:
        _CASES_CACHE = _load_cases()
    return _CASES_CACHE.get(case_id)

def pick_majority_case_id(docs: List[Dict[str, Any]]) -> Optional[str]:
    """retrieved 상위 문서들에서 가장 많이 등장한 사고유형ID 선택(동률이면 첫 등장 우선)"""
    freq: Dict[str, int] = {}
    order: List[str] = []
    for d in docs:
        cid = d.get("사고유형ID")
        if not cid: 
            continue
        if cid not in freq:
            freq[cid] = 0
            order.append(cid)
        freq[cid] += 1
    if not freq:
        return None
    # 최빈값, 동률 시 먼저 나온 case_id
    best = max(freq.items(), key=lambda x: (x[1], -order.index(x[0])))
    return best[0]

def extract_statutes(case_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """적용법조항 배열만 깔끔히 추출"""
    items = case_obj.get("적용법조항") or []
    out = []
    for it in items:
        name = (it.get("조문명") or "").strip()
        core = (it.get("핵심내용") or "").strip()
        if name or core:
            out.append({"조문명": name, "핵심내용": core})
    return out

def extract_precedents(case_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """참고판례 배열만 깔끔히 추출"""
    items = case_obj.get("참고판례") or []
    out = []
    for it in items:
        src = (it.get("출처") or "").strip()
        gist = (it.get("판결요지") or "").strip()
        if src or gist:
            out.append({"출처": src, "판결요지": gist})
    return out