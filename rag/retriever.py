# rag/retriever.py
import pickle, pathlib, sys, re
from typing import List, Dict, Any, Optional
import faiss
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

VEC_DIR = ROOT / "data" / "vector_index"
MODEL_TXT = VEC_DIR / "model.txt"

_A_PAT = re.compile(r"\bA\b|A측|A 차량|A차량")
_B_PAT = re.compile(r"\bB\b|B측|B 차량|B차량")

class Retriever:
    def __init__(self, vec_dir: pathlib.Path = VEC_DIR):
        self.vec_dir = vec_dir
        self.index = faiss.read_index(str(vec_dir / "faiss.index"))
        self.meta: List[Dict[str, Any]] = pickle.loads((vec_dir / "meta.pkl").read_bytes())
        model_name = MODEL_TXT.read_text(encoding="utf-8").strip()
        self.model = SentenceTransformer(model_name)

    def _who_target(self, m: Dict[str, Any]) -> Optional[str]:
        # 메타에 명시된 타겟이 있으면 우선 사용
        t = m.get("adjust_target")
        if t in ("A", "B"):
            return t
        # 텍스트에서 휴리스틱
        txt = (m.get("text") or "") + " " + (m.get("title") or "") + " " + (m.get("section") or "")
        if _A_PAT.search(txt) and not _B_PAT.search(txt):
            return "A"
        if _B_PAT.search(txt) and not _A_PAT.search(txt):
            return "B"
        return None

    def search(self, query: str, top_k: int = 8, section_bonus: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        section_bonus = section_bonus or {}
        emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(emb, max(top_k * 2, 16))  # 여유 추출

        out: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = dict(self.meta[idx])
            sec = m.get("section", "")
            bonus = 0.0

            # 기존 고정 가중치
            if "과실비율조정예시" in sec: bonus += 0.06
            if "적용법조항" in sec:     bonus += 0.04
            if "참고판례" in sec:       bonus += 0.04

            # prior 기반 섹션 가중치
            bonus += float(section_bonus.get("법조문", 0.0))   if "적용법조항" in sec else 0.0
            bonus += float(section_bonus.get("판례", 0.0))     if "참고판례" in sec else 0.0
            if "과실비율조정예시" in sec:
                # 공통 보정 + A/B 측면 보정
                bonus += float(section_bonus.get("조정예시_공통", 0.0))
                tgt = self._who_target(m)
                if tgt == "A":
                    bonus += float(section_bonus.get("조정예시_A측", 0.0))
                elif tgt == "B":
                    bonus += float(section_bonus.get("조정예시_B측", 0.0))

            m["score"] = float(score + bonus)
            out.append(m)

        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]