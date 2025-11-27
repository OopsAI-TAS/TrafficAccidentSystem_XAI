# xai/batch_ig.py

import json
import os
import torch
from pathlib import Path
import pandas as pd

from ig_explain import pretty_ig, load_model_and_tokenizer, aggregate_feature_attributions


RESULT_DIR = Path("xai/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = "train/valid.jsonl"  # test셋 있으면 바꿔도 됨


def split_feature_key(feat_key, base_keys):
    """
    예: feat_key = '유형코드=214', base_keys = ['유형코드', '장소', ...]
    -> ('유형코드', '214')
    """
    if not isinstance(feat_key, str):
        return None, None

    # 길이가 긴 key(장소특징)를 먼저 매칭하도록 정렬
    for base in sorted(base_keys, key=len, reverse=True):
        if feat_key.startswith(base):
            rest = feat_key[len(base):].strip()
            if rest.startswith("="):
                rest = rest[1:].strip()
            return base, rest
    return None, None


############################################################
# 1) 전체 IG 계산 → CSV 저장
############################################################

def run_batch_ig():
    print("=== Running batch IG on dataset ===")

    type_rows = []
    value_rows = []

    tok, model = load_model_and_tokenizer()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    base_keys = ["유형코드", "장소", "장소특징", "A진행", "B진행"]
    alias_map = {
        "유형코드": "feat_acc_type",
        "장소": "feat_place",
        "장소특징": "feat_place_feat",
        "A진행": "feat_A_move",
        "B진행": "feat_B_move",
    }

    for idx, line in enumerate(lines):
        r = json.loads(line)
        text, true_A = r["text"], r["A"]

        # 1) pred_A 계산
        enc = tok(
            text,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        input_ids = enc["input_ids"].to(model.bert.device)
        attention_mask = enc["attention_mask"].to(model.bert.device)

        with torch.no_grad():
            predA = model(input_ids=input_ids, attention_mask=attention_mask).item()

        # 2) IG 토큰/점수
        (raw_tokens, raw_scores), (merged_tokens, merged_scores) = pretty_ig(
            text, n_steps=50
        )

        # 3) feature-level attribution
        feat_scores = aggregate_feature_attributions(merged_tokens, merged_scores)

        # ------------------------------
        # 타입별 요약
        # ------------------------------
        type_scores = {k: 0.0 for k in base_keys}

        for fk, fv in feat_scores.items():
            ftype, fval = split_feature_key(fk, base_keys)
            if ftype is None:
                continue

            type_scores[ftype] += fv

            value_rows.append({
                "sample_idx": idx,
                "true_A": true_A,
                "pred_A": predA,
                "feature_type": ftype,
                "feature_alias": alias_map[ftype],
                "feature_value": fval,
                "ig_score": fv,
            })

        row_type = {
            "true_A": true_A,
            "pred_A": predA,
            "feat_acc_type": type_scores["유형코드"],
            "feat_place": type_scores["장소"],
            "feat_place_feat": type_scores["장소특징"],
            "feat_A_move": type_scores["A진행"],
            "feat_B_move": type_scores["B진행"],
        }
        type_rows.append(row_type)

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(lines)}")

    # 타입 단위 저장
    df_type = pd.DataFrame(type_rows)
    df_type.to_csv(RESULT_DIR / "ig_results.csv", index=False)
    print("Saved IG type-level results →", RESULT_DIR / "ig_results.csv")

    # value 단위 저장
    df_value = pd.DataFrame(value_rows)
    df_value.to_csv(RESULT_DIR / "ig_value_results.csv", index=False)
    print("Saved IG value-level results →", RESULT_DIR / "ig_value_results.csv")

    return df_type, df_value


############################################################
# 2) 자동 분석 레포트 생성 (텍스트만)
############################################################

def write_analysis(df):
    print("=== Writing analysis report ===")

    feat_cols = ["feat_acc_type", "feat_place", "feat_place_feat", "feat_A_move", "feat_B_move"]

    text = []
    text.append("=== XAI IG Feature Analysis Report ===\n")

    # 평균 기여도
    text.append("\n[Feature Mean Attribution]\n")
    means = df[feat_cols].mean().sort_values(ascending=False)
    text.append(str(means))

    # 절댓값 기준 중요도
    text.append("\n\n[Feature |mean(|attr|)|]\n")
    abs_means = df[feat_cols].abs().mean().sort_values(ascending=False)
    text.append(str(abs_means))

    # pred_A 관련성
    corr_pred = df[["pred_A"] + feat_cols].corr()["pred_A"].sort_values(ascending=False)
    text.append("\n\n[Correlation with pred_A]\n")
    text.append(str(corr_pred))

    # true_A 관련성
    corr_true = df[["true_A"] + feat_cols].corr()["true_A"].sort_values(ascending=False)
    text.append("\n\n[Correlation with true_A]\n")
    text.append(str(corr_true))

    path = RESULT_DIR / "analysis.txt"
    with open(path, "w") as f:
        f.write("\n".join(text))

    print("Saved analysis.txt →", path)


############################################################
# MAIN
############################################################

if __name__ == "__main__":
    df_type, df_value = run_batch_ig()
    write_analysis(df_type)

    print("\n=== DONE ===")
    print("Check folder:", RESULT_DIR)