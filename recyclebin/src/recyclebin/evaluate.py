"""Evaluate on the held-out test split.
Usage: python -m recyclebin.evaluate --model baseline|mobilenetv2
Writes predictions.csv, metrics.json, confusion_matrix.csv, classification_report.txt
"""
import argparse, json
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .config import load_config
from .data import build_splits, make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "mobilenetv2"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes = cfg["classes"]
    df = build_splits(cfg)
    test = df[df.split == "test"].reset_index(drop=True)
    model = tf.keras.models.load_model(Path(cfg["output"]["models_dir"]) / f"{args.model}.keras")

    probs = model.predict(make_dataset(test, cfg))
    y_true = test["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    y_pred = probs.argmax(1)

    out = Path(cfg["output"]["results_dir"]) / args.model; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": test["path"], "true": [classes[i] for i in y_true],
                  "pred": [classes[i] for i in y_pred], "confidence": probs.max(1)}
                 ).to_csv(out / "predictions.csv", index=False)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {"n": int(len(test)), "correct": int((y_true == y_pred).sum()),
               "accuracy": accuracy_score(y_true, y_pred),
               "precision_macro": p, "recall_macro": r, "f1_macro": f}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, digits=3))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
