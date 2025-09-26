# -*- coding: utf-8 -*-
import json, pickle, pathlib, sys, os
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer

# --- 모듈 경로 보정 ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ingestion.text_normalizer import normalize_ko

DATA = ROOT / "data" / "law_json" / "cases.clean.jsonl"
OUTD = ROOT / "data" / "vector_index"; OUTD.mkdir(parents=True, exist_ok=True)
SAMPLE_INPUT = ROOT / "samples" / "input.json"

def load_video_meta(p: pathlib.Path) -> Dict[str, Any]:
    """samples/input.json에서 video_name, video_date를 가져온다(있으면)."""
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        v = obj.get("video", {}) or {}
        return {
            "video_name": v.get("video_name"),
            "video_date": v.get("video_date"),
        }
    except Exception:
        return {}

def load_cases(p: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            rows.append(json.loads(line))
    return rows

def chunk_case(case: Dict[str, Any], vmeta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """한 케이스를 여러 검색 청크로 분해. 각 청크에 video_name/date 메타 포함."""
    cid = case.get("사고유형ID",""); title = case.get("사고유형명","")
    base = case.get("기본과실비율") or {}
    a = base.get("A차량"); b = base.get("B차량")
    chunks: List[Dict[str, Any]] = []

    def add(section: str, text: str):
        text = normalize_ko(text or "")
        if not text:
            return
        chunks.append({
            "사고유형ID": cid,
            "사고유형명": title,
            "section": section,
            "A_base": a,
            "B_base": b,
            "text": text,
            # ← 여기서 비디오 메타를 함께 저장
            "video_name": vmeta.get("video_name"),
            "video_date": vmeta.get("video_date"),
        })

    add("사고상황설명", case.get("사고상황설명"))
    add("기본과실비율_비율설명", base.get("비율설명"))

    # 과실비율 조정예시 묶어서 한 청크
    adj = case.get("과실비율조정예시") or []
    if adj:
        adj_txt = "\n".join([f"- {x.get('대상')} | {x.get('가산사유')} | {x.get('조정값')}" for x in adj])
        add("과실비율조정예시", adj_txt)

    # 법/판례 각각 개별 청크
    for law in case.get("적용법조항") or []:
        add("적용법조항", f"{law.get('조문명','')} – {law.get('핵심내용','')}")
    for pre in case.get("참고판례") or []:
        add("참고판례", f"{pre.get('출처','')} – {pre.get('판결요지','')}")

    add("설명요약", case.get("설명요약"))
    return chunks

def main(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
    # 비디오 메타 로드
    vmeta = load_video_meta(SAMPLE_INPUT)

    # 케이스 로드 및 청크 생성(비디오 메타 포함)
    cases = load_cases(DATA)
    corpus = [c for case in cases for c in chunk_case(case, vmeta)]
    print(f"[chunks] {len(corpus)}")

    # 임베딩 & 인덱싱
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in corpus]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # 저장 (meta에 video_name/date 포함됨)
    faiss.write_index(index, str(OUTD/"faiss.index"))
    with open(OUTD/"meta.pkl","wb") as f: 
        pickle.dump(corpus, f)
    (OUTD/"model.txt").write_text(model_name, encoding="utf-8")
    print(f"[ok] index saved → {OUTD}")

if __name__ == "__main__":
    main()