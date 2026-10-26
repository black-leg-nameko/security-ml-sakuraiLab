"""Draw the explanatory figures used by the seminar notebooks."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "figures"
INK = "#183247"
TEAL = "#147d83"
ORANGE = "#b56327"
PALE = "#e9f3f4"


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def box(ax, center, width, height, label, color=PALE, fontsize=11):
    x, y = center
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=color,
            edgecolor=INK,
            linewidth=1,
        )
    )
    ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=fontsize)


def arrow(ax, start, end, color=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=color))


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT / f"{name}.png", dpi=170, facecolor="white", metadata={"Software": "Matplotlib"}
    )
    plt.close(fig)


def workflow():
    fig, ax = canvas(13, 3.5)
    labels = [
        "収集・ラベル付け",
        "読み込み・前処理",
        "探索・\n特徴量の検討",
        "訓練・評価",
        "運用・監視",
    ]
    centers = np.linspace(0.105, 0.895, 5)
    for x, label in zip(centers, labels, strict=True):
        box(ax, (x, 0.62), 0.16, 0.27, label)
    for left, right in zip(centers[:-1], centers[1:], strict=True):
        arrow(ax, (left + 0.087, 0.62), (right - 0.087, 0.62))
    ax.plot([centers[3], centers[3], centers[2]], [0.47, 0.24, 0.24], color=ORANGE)
    arrow(ax, (centers[2], 0.24), (centers[2], 0.47), ORANGE)
    ax.text(0.60, 0.11, "訓練と評価を繰り返して見直す", ha="center", color=ORANGE)
    ax.text(
        0.5,
        0.91,
        "モデル開発の工程",
        ha="center",
        weight="bold",
        color=INK,
        fontsize=14,
    )
    save(fig, "01_workflow")


def split_cv():
    fig, ax = canvas(12, 5)
    for x, width, color, label in [
        (0.05, 0.72, TEAL, "訓練用 80%"),
        (0.77, 0.18, INK, "テスト用 20%"),
    ]:
        ax.add_patch(
            Rectangle((x, 0.77), width, 0.12, facecolor=color, edgecolor="white", linewidth=2)
        )
        ax.text(x + width / 2, 0.83, label, ha="center", va="center", color="white", weight="bold")
    ax.text(0.41, 0.68, "訓練部分の内側で5分割交差検証", ha="center", color=INK)
    for row in range(5):
        y = 0.54 - 0.09 * row
        ax.text(0.035, y + 0.03, f"{row + 1}", ha="right", va="center", color=INK)
        for col in range(5):
            x = 0.05 + col * 0.144
            color = ORANGE if col == row else TEAL
            ax.add_patch(
                Rectangle((x, y), 0.144, 0.06, facecolor=color, edgecolor="white", linewidth=2)
            )
            ax.text(
                x + 0.072,
                y + 0.03,
                "CV評価" if col == row else "学習",
                ha="center",
                va="center",
                color="white",
            )
    box(ax, (0.86, 0.39), 0.15, 0.23, "正解率・\n混同行列の評価", color="#edf0f4")
    ax.text(0.41, 0.06, "5回の正解率を平均して設定を比較する", ha="center", color=TEAL)
    ax.text(
        0.5,
        0.97,
        "本書の分割：訓練80%・テスト20%",
        ha="center",
        weight="bold",
        color=INK,
        fontsize=13,
    )
    save(fig, "02_split_cv")


def sigmoid():
    fig, ax = plt.subplots(figsize=(8.5, 4.1), layout="constrained")
    z = np.linspace(-6, 6, 500)
    ax.plot(z, 1 / (1 + np.exp(-z)), color=TEAL, linewidth=2.5)
    for threshold, color in [(0.5, INK)]:
        boundary = np.log(threshold / (1 - threshold))
        ax.axhline(threshold, color=color, linestyle="--", alpha=0.7)
        ax.vlines(boundary, 0, threshold, color=color, linestyle=":")
        ax.scatter([boundary], [threshold], color=color, zorder=3)
        ax.text(3.4, threshold + 0.035, f"閾値 = {threshold}", color=color)
    ax.set(
        xlim=(-6, 6),
        ylim=(0, 1.05),
        xlabel=r"重み付き和 $z = w^T x + b$",
        ylabel=r"スコア $\sigma(z)$",
        title="ロジスティック回帰のシグモイド",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    save(fig, "03_sigmoid")


def tree():
    fig, ax = canvas(10, 4.7)
    box(ax, (0.5, 0.85), 0.24, 0.13, r"$x_A \leq -0.5$ ?")
    box(ax, (0.23, 0.53), 0.25, 0.14, "葉：スコア高", color="#fbefe5")
    box(ax, (0.73, 0.53), 0.24, 0.13, r"$x_B \leq 0.5$ ?")
    box(ax, (0.58, 0.20), 0.24, 0.14, "葉：スコア中")
    box(ax, (0.87, 0.20), 0.24, 0.14, "葉：スコア低")
    for start, end, label in [
        ((0.44, 0.775), (0.25, 0.61), "はい"),
        ((0.56, 0.775), (0.71, 0.605), "いいえ"),
        ((0.69, 0.455), (0.59, 0.28), "はい"),
        ((0.77, 0.455), (0.86, 0.28), "いいえ"),
    ]:
        arrow(ax, start, end)
        ax.text(
            (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.03, label, color=INK, ha="center"
        )
    ax.text(
        0.5,
        0.03,
        "説明用の仮の条件とスコア",
        ha="center",
        color=INK,
        fontsize=10,
    )
    save(fig, "04_tree")


def tfidf():
    documents = ["meeting agenda", "meeting report", "prize prize"]
    counts = CountVectorizer().fit_transform(documents).toarray()
    vectorizer = TfidfVectorizer()
    values = vectorizer.fit_transform(documents).toarray()
    words = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    for ax, data, title, fmt in zip(
        axes,
        [counts, values],
        ["単語の出現回数", "TF-IDF（L2正規化後）"],
        [".0f", ".2f"],
        strict=True,
    ):
        ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max(), aspect="auto")
        ax.set(
            xticks=range(len(words)),
            xticklabels=words,
            yticks=range(3),
            yticklabels=documents,
            title=title,
        )
        for row in range(3):
            for col in range(len(words)):
                ax.text(
                    col,
                    row,
                    format(data[row, col], fmt),
                    ha="center",
                    va="center",
                    color="white" if data[row, col] > data.max() * 0.6 else INK,
                )
        ax.tick_params(length=0)
    save(fig, "05_tfidf")


def main():
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 11})
    for draw in (workflow, split_cv, sigmoid, tree, tfidf):
        draw()


if __name__ == "__main__":
    main()
