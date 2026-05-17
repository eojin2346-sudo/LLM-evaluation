# =============================================================
# visualize.py — 결과 시각화
# =============================================================

import json
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import platform
import llm_config

# 한글 폰트 설정
if platform.system() == "Darwin":
    matplotlib.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = ["#E8593C", "#3B8BD4", "#2CA02C", "#9467BD", "#FF7F0E"]

def plot_single_heatmap(df: pd.DataFrame):
    """모델 × 전략별 평균 점수 히트맵"""
    pivot = df.pivot_table(values="score", index="model", columns="strategy", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=1, vmax=10)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticklabels(pivot.index, fontsize=11)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.1f}",
                    ha="center", va="center", fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label="평균 점수 (max 10)")
    ax.set_title("모델 × 프롬프트 전략별 응답 품질 (Single Judge)", fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(llm_config.RESULTS_DIR, "heatmap_single.png")
    plt.savefig(path, dpi=150)
    print(f"저장: {path}")
    plt.show()

def plot_pairwise_bar(df: pd.DataFrame):
    """모델별 Pairwise 승률 막대 그래프"""
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    win_rates = []

    for model in models:
        mask  = (df["model_a"] == model) | (df["model_b"] == model)
        total = mask.sum()
        wins  = (df["winner"] == model).sum()
        win_rates.append(wins / total * 100 if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, win_rates, color=COLORS[:len(models)], edgecolor="white", linewidth=1.5)

    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("승률 (%)", fontsize=11)
    ax.set_title("모델별 Pairwise 승률", fontsize=13)
    ax.set_ylim(0, 100)
    ax.axhline(y=33.3, color="gray", linestyle="--", alpha=0.5, label="기준선 (33%)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(llm_config.RESULTS_DIR, "pairwise_winrate.png")
    plt.savefig(path, dpi=150)
    print(f"저장: {path}")
    plt.show()

def plot_strategy_comparison(df: pd.DataFrame):
    """전략별 점수 분포 박스플롯"""
    strategies = df["strategy"].unique()
    data = [df[df["strategy"] == s]["score"].values for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, labels=strategies, patch_artist=True)

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("점수 (10점 만점)", fontsize=11)
    ax.set_title("프롬프트 전략별 점수 분포", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(llm_config.RESULTS_DIR, "strategy_boxplot.png")
    plt.savefig(path, dpi=150)
    print(f"저장: {path}")
    plt.show()

def run():
    with open(llm_config.SCORES_PATH, "r", encoding="utf-8") as f:
        df_single = pd.DataFrame(json.load(f))
    with open(llm_config.PAIRWISE_PATH, "r", encoding="utf-8") as f:
        df_pairwise = pd.DataFrame(json.load(f))

    os.makedirs(llm_config.RESULTS_DIR, exist_ok=True)
    plot_single_heatmap(df_single)
    plot_pairwise_bar(df_pairwise)
    plot_strategy_comparison(df_single)

if __name__ == "__main__":
    run()
