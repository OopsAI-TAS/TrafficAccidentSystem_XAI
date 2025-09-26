import os, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.text_normalizer import normalize_ko

def norm_obj(o):
    if isinstance(o, str): return normalize_ko(o)
    if isinstance(o, list): return [norm_obj(x) for x in o]
    if isinstance(o, dict): return {k: norm_obj(v) for k, v in o.items()}
    return o

src = sys.argv[1]
dst = sys.argv[2]
with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip(): continue
        obj = json.loads(line)
        obj = norm_obj(obj)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"cleaned → {dst}")