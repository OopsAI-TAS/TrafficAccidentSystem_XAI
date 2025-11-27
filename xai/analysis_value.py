# xai/analysis_value.py

import json
import re
from pathlib import Path

import pandas as pd
RESULT_DIR = Path("xai/results")
VALUE_PATH = RESULT_DIR / "ig_value_results.csv"
PROGRESS_MAP_PATH = Path("xai/progress_maps.json")

# -------------------------------------------------------------------
# 1) A/B 진행코드 → 카테고리 역매핑 테이블 생성
# -------------------------------------------------------------------
def load_progress_inv():
    mp = json.load(open(PROGRESS_MAP_PATH, "r", encoding="utf-8"))
    inv = {"A진행": {}, "B진행": {}}

    for cat, codes in mp["vehicle_a_progress_info"].items():
        for c in codes:
            if c is not None:
                inv["A진행"][int(c)] = cat

    for cat, codes in mp["vehicle_b_progress_info"].items():
        for c in codes:
            if c is not None:
                inv["B진행"][int(c)] = cat

    return inv

# -------------------------------------------------------------------
# 2) '직진(코드=6)' → 6 뽑기
# -------------------------------------------------------------------
def extract_code(value_str: str):
    if not isinstance(value_str, str):
        return None
    m = re.search(r"코드=(\d+)", value_str)
    if m:
        return int(m.group(1))
    return None

# -------------------------------------------------------------------
# 3) IG CSV 로드 + A/B 진행 카테고리 매핑 + agg_value 생성
# -------------------------------------------------------------------
def load_value_df():
    df = pd.read_csv(VALUE_PATH)
    progress_inv = load_progress_inv()

    mask_ab = df["feature_type"].isin(["A진행", "B진행"])

    df.loc[mask_ab, "code"] = df.loc[mask_ab, "feature_value"].apply(extract_code)

    def map_to_cat(row):
        code = row["code"]
        if pd.isna(code):
            return "UNKNOWN"
        return progress_inv[row["feature_type"]].get(int(code), "UNKNOWN")

    df.loc[mask_ab, "category"] = df.loc[mask_ab].apply(map_to_cat, axis=1)

    # 분석용 최종 Key
    df["agg_value"] = df["feature_value"]
    df.loc[mask_ab, "agg_value"] = df.loc[mask_ab, "category"]

    print("Loaded:", VALUE_PATH)
    print(df.head())
    return df

# -------------------------------------------------------------------
# 4) (feature_type, agg_value)별 평균 IG + CSV 저장
# -------------------------------------------------------------------
def compute_mean_csv(df):
    out_path = RESULT_DIR / "ig_value_summary.csv"

    g = (
        df.groupby(["feature_type", "agg_value"])["ig_score"]
        .mean()
        .reset_index()
        .sort_values("ig_score", ascending=False)
    )

    g.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary CSV → {out_path}")

    return g

# -------------------------------------------------------------------
# 5) feature_type별 개별 CSV 저장
# -------------------------------------------------------------------
def save_per_feature_type(df):
    g = (
        df.groupby(["feature_type", "agg_value"])["ig_score"]
        .mean()
        .reset_index()
        .sort_values("ig_score", ascending=False)
    )

    for t in sorted(df["feature_type"].unique()):
        out_path = RESULT_DIR / f"ig_value_summary_by_type_{t}.csv"
        g_t = g[g["feature_type"] == t]
        g_t.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved → {out_path}")

# -------------------------------------------------------------------
# 6) main
# -------------------------------------------------------------------
def main():
    df = load_value_df()

    compute_mean_csv(df)
    save_per_feature_type(df)

    print("\n=== DONE 13주차 (CSV Export Complete, NO PLOTS) ===")
    print("See results in:", RESULT_DIR)


if __name__ == "__main__":
    main()
