from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root 기준 맞춰 필요시 조정
# 진행방향 매핑(코드→카테고리) JSON 경로: infer.py에서 바꾼 경로와 동일
PROGRESS_MAP_PATHS = [
    ROOT / "train" / "data" / "mappings" / "progress_maps.json",  # 1순위 (네가 방금 쓴 경로)
    ROOT / "data"  / "mappings" / "progress_maps.json",           # 2순위(백업 위치)
]
def get_progress_map_path() -> Path:
    # 통합 맵의 실제 위치로 지정
    return ROOT / "train" / "data" / "mappings" / "progress_maps.json"