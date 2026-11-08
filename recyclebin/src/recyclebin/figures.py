"""Regenerate paper figures from results files (no TensorFlow needed).
Usage: python -m recyclebin.figures [confusion_csv] [metrics_json]
Defaults use the values reported in the team presentation (results/reported_*).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9})


def fig_metrics(metrics, out):
    names = ["Accuracy", "Precision (macro)", "Recall (macro)", "Macro F1"]
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    b = [metrics["baseline"][k] for k in keys]; t = [metrics["mobilenetv2"][k] for k in keys]
    x = np.arange(4); w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 2.7), dpi=220)
    r1 = ax.bar(x - w / 2, b, w, label="Baseline CNN (from scratch)", color="#9a9a9a", edgecolor="black", linewidth=.5)
    r2 = ax.bar(x + w / 2, t, w, label="MobileNetV2 (ImageNet init.)", color="#2f6f4f", edgecolor="black", linewidth=.5)
    for r in list(r1) + list(r2):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .015, f"{r.get_height():.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(.5, 1.16))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_confusion(cm_df, out):
    cls = list(cm_df.index); cm = cm_df.to_numpy()
    tp = np.diag(cm); P = tp / cm.sum(0); R = tp / cm.sum(1); F = 2 * P * R / (P + R)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.2), dpi=220, gridspec_kw={"width_ratios": [1.05, 1]})
    a.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max() * 1.2)
    for i in range(len(cls)):
        for j in range(len(cls)):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                   color="white" if cm[i, j] > cm.max() * .5 else "black")
    a.set_xticks(range(len(cls))); a.set_yticks(range(len(cls)))
    a.set_xticklabels(cls, rotation=45, ha="right"); a.set_yticklabels(cls)
    a.set_xlabel("Predicted class"); a.set_ylabel("True class")
    a.set_title(f"(a) Confusion matrix, n = {cm.sum()}", fontsize=9)
    y = np.arange(len(cls)); h = .27
    b.barh(y - h, P, h, label="Precision", color="#9a9a9a", edgecolor="black", linewidth=.4)
    b.barh(y, R, h, label="Recall", color="#5aa17f", edgecolor="black", linewidth=.4)
    b.barh(y + h, F, h, label="F1", color="#2f6f4f", edgecolor="black", linewidth=.4)
    b.set_yticks(y); b.set_yticklabels(cls); b.invert_yaxis(); b.set_xlim(0, 1.05); b.set_xlabel("Score")
    b.set_title("(b) Per-class precision, recall, F1", fontsize=9); b.legend(frameon=False, fontsize=7.5, loc="lower right")
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


if __name__ == "__main__":
    cm_path = sys.argv[1] if len(sys.argv) > 1 else "results/reported_confusion_matrix_mobilenetv2.csv"
    mt_path = sys.argv[2] if len(sys.argv) > 2 else "results/reported_metrics.json"
    out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)
    fig_metrics(json.load(open(mt_path)), out / "fig1_metrics.png")
    fig_confusion(pd.read_csv(cm_path, index_col=0), out / "fig2_confusion_and_per_class.png")
    print("Figures written to", out)
