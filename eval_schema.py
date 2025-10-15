from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class EvalCriterion(BaseModel):
    name: str
    weight: int                 # 0~100, 합계 100
    score: int                  # 0~100 (가중치 이전 원점수)
    comments: str               # 간결한 한국어 코멘트
    evidence: List[str] = []    # 문서 내 근거 구절(요약/발췌)

class EvalFlags(BaseModel):
    hallucination: bool = False
    missing_fields: List[str] = []
    template_violations: List[str] = []
    math_errors: List[str] = []

class Extracted(BaseModel):
    final_ratio: Dict[str, int] = {}      # {"A":60,"B":40} 식
    cited_laws: List[str] = []
    cited_precedents: List[str] = []

class EvalResult(BaseModel):
    total_score: int                       # 0~100 (가중 합산)
    criteria: List[EvalCriterion]
    flags: EvalFlags
    suggestions: List[str] = []            # 개선 제안
    extracted: Extracted
