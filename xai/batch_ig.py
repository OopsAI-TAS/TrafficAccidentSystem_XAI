# xai/batch_ig.py

import json
import os
import torch
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 서버 환경에서 GUI 오류 방지
import matplotlib.pyplot as plt
import seaborn as sns

from ig_explain import pretty_ig, load_model_and_tokenizer, aggregate_feature_attributions

RESULT_DIR = Path("xai/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = "train/valid.jsonl"  # test셋 있으면 바꿔도 됨


#############################################
# 1) 전체 IG 계산 → CSV 저장
#############################################

def run_batch_ig():
    print("=== Running batch IG on dataset ===")

    rows = []
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

        # 3) feature-level attribution dict (예: "유형코드=214": 0.02, ...)
        feat_scores = aggregate_feature_attributions(merged_tokens, merged_scores)

        # 4) 타입별(유형코드 / 장소 / ...)로 한 번 더 묶기
        type_scores = {k: 0.0 for k in base_keys}
        for fk, fv in feat_scores.items():
            for base in base_keys:
                if fk.startswith(base):
                    type_scores[base] = fv

        row = {
            "true_A": true_A,
            "pred_A": predA,
            "feat_acc_type": type_scores["유형코드"],
            "feat_place": type_scores["장소"],
            "feat_place_feat": type_scores["장소특징"],
            "feat_A_move": type_scores["A진행"],
            "feat_B_move": type_scores["B진행"],
        }
        rows.append(row)

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(lines)}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "ig_results.csv", index=False)
    print("Saved IG results →", RESULT_DIR / "ig_results.csv")
    return df


#############################################
# 2) 그래프 자동 생성
#############################################

def draw_plots(df):
    print("=== Drawing plots ===")

    feat_cols = ["feat_acc_type", "feat_place", "feat_place_feat", "feat_A_move", "feat_B_move"]

    # Boxplot
    plt.figure(figsize=(9,6))
    sns.boxplot(data=df[feat_cols])
    plt.title("IG Attribution Distribution")
    plt.savefig(RESULT_DIR / "boxplot.png")
    plt.close()

    # Violin
    plt.figure(figsize=(9,6))
    sns.violinplot(data=df[feat_cols])
    plt.title("IG Attribution Violin Plot")
    plt.savefig(RESULT_DIR / "violin.png")
    plt.close()

    # Scatter: pred_A vs 각 feature
    for feat in feat_cols:
        plt.figure(figsize=(8,5))
        sns.scatterplot(x=df["pred_A"], y=df[feat])
        plt.title(f"pred_A vs {feat}")
        plt.xlabel("pred_A")
        plt.ylabel(feat)
        plt.savefig(RESULT_DIR / f"scatter_predA_vs_{feat}.png")
        plt.close()

    # Scatter: true_A vs pred_A
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["true_A"], y=df["pred_A"])
    plt.title("true_A vs pred_A")
    plt.savefig(RESULT_DIR / "scatter_trueA_vs_predA.png")
    plt.close()

    print("Saved all plots!")


#############################################
# 3) 자동 분석 레포트 생성
#############################################

def write_analysis(df):
    print("=== Writing analysis report ===")

    feat_cols = ["feat_acc_type", "feat_place", "feat_place_feat", "feat_A_move", "feat_B_move"]

    text = []
    text.append("=== XAI IG Feature Analysis Report ===\n")

    # 1) 평균 기여도 순위
    text.append("\n[Feature Mean Attribution]\n")
    means = df[feat_cols].mean().sort_values(ascending=False)
    text.append(str(means))

    # 2) 절댓값 기준 중요도
    text.append("\n\n[Feature |mean(|attr|)|]\n")
    abs_means = df[feat_cols].abs().mean().sort_values(ascending=False)
    text.append(str(abs_means))

    # 3) pred_A 관련성
    corr_pred = df[["pred_A"] + feat_cols].corr()["pred_A"].sort_values(ascending=False)
    text.append("\n\n[Correlation with pred_A]\n")
    text.append(str(corr_pred))

    # 4) true_A 관련성
    corr_true = df[["true_A"] + feat_cols].corr()["true_A"].sort_values(ascending=False)
    text.append("\n\n[Correlation with true_A]\n")
    text.append(str(corr_true))

    path = RESULT_DIR / "analysis.txt"
    with open(path, "w") as f:
        f.write("\n".join(text))

    print("Saved analysis.txt →", path)


#############################################
# MAIN
#############################################

if __name__ == "__main__":
    df = run_batch_ig()
    draw_plots(df)
    write_analysis(df)

    print("\n=== DONE ===")
    print("Check folder:", RESULT_DIR)
