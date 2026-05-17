# =============================================================
# evaluate.py — 결과 집계 및 요약 출력
# =============================================================

import json
import pandas as pd
import llm_config

def load_scores() -> pd.DataFrame:
    with open(llm_config.SCORES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_pairwise() -> pd.DataFrame:
    with open(llm_config.PAIRWISE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def summarize_single(df: pd.DataFrame):
    print("\n" + "="*60)
    print("Single Judge 결과 — 모델 × 전략별 평균 점수 (10점 만점)")
    print("="*60)
    pivot = df.pivot_table(values="score", index="model", columns="strategy", aggfunc="mean")
    pivot["평균"] = pivot.mean(axis=1)
    print(pivot.round(2).to_string())

    print("\n--- 카테고리별 평균 점수 ---")
    cat = df.groupby(["model", "category"])["score"].mean().unstack()
    print(cat.round(2).to_string())

def summarize_pairwise(df: pd.DataFrame):
    print("\n" + "="*60)
    print("Pairwise Judge 결과 — 모델별 승/패/무")
    print("="*60)

    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    records = []

    for model in models:
        wins  = len(df[df["winner"] == model])
        ties  = len(df[df["winner"] == "tie"])
        total = len(df[(df["model_a"] == model) | (df["model_b"] == model)])
        losses = total - wins - ties
        records.append({"모델": model, "승": wins, "패": losses, "무": ties, "총": total})

    result_df = pd.DataFrame(records).set_index("모델")
    print(result_df.to_string())

def run():
    df_single   = load_scores()
    df_pairwise = load_pairwise()

    summarize_single(df_single)
    summarize_pairwise(df_pairwise)

    print("\n결과 파일:")
    print(f"  Single  : {llm_config.SCORES_PATH}")
    print(f"  Pairwise: {llm_config.PAIRWISE_PATH}")

if __name__ == "__main__":
    run()
