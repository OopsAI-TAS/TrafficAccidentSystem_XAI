# # -*- coding: utf-8 -*-
# import pickle, pathlib, faiss
# from typing import List, Dict, Any
# from sentence_transformers import SentenceTransformer

# ROOT = pathlib.Path(__file__).resolve().parents[1]
# OUTD = ROOT / "data" / "vector_index"

# def load_retriever():
#     index = faiss.read_index(str(OUTD/"faiss.index"))
#     meta = pickle.load(open(OUTD/"meta.pkl","rb"))
#     model_name = (OUTD/"model.txt").read_text(encoding="utf-8").strip()
#     model = SentenceTransformer(model_name)
#     return index, meta, model

# def search(index, meta, model, query: str, topk=8) -> List[Dict[str, Any]]:
#     qv = model.encode([query], normalize_embeddings=True)
#     D, I = index.search(qv, topk)
#     return [meta[i] for i in I[0] if i >= 0]

# def format_context(ctxs):
#     lines = []
#     a = next((c.get("A_base") for c in ctxs if c.get("A_base") is not None), "")
#     b = next((c.get("B_base") for c in ctxs if c.get("B_base") is not None), "")
#     for c in ctxs:
#         lines.append(f"[{c['사고유형ID']}][{c['section']}] {c['text']}")
#     return "\n".join(lines), a, b
# rag/retriever.py
# -*- coding: utf-8 -*-
import pickle, pathlib, sys
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

VEC_DIR = ROOT / "data" / "vector_index"
MODEL_TXT = VEC_DIR / "model.txt"

class Retriever:
    def __init__(self, vec_dir: pathlib.Path = VEC_DIR):
        self.vec_dir = vec_dir
        self.index = faiss.read_index(str(vec_dir / "faiss.index"))
        self.meta: List[Dict[str, Any]] = pickle.loads((vec_dir / "meta.pkl").read_bytes())
        model_name = MODEL_TXT.read_text(encoding="utf-8").strip()
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(emb, max(top_k*2, 16))  # 여유로 뽑고
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            m = dict(self.meta[idx])
            sec = m.get("section","")
            bonus = 0.0
            if "과실비율조정예시" in sec: bonus += 0.06
            if "적용법조항" in sec:     bonus += 0.04
            if "참고판례" in sec:       bonus += 0.04
            m["score"] = float(score + bonus)
            out.append(m)
        # 보정점수로 재정렬 후 상위 top_k
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]