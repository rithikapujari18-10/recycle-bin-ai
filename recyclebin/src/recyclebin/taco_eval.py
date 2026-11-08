"""Spot-check on real-world TACO photos.

Input CSV columns: path,true_label  (true_label must be one of the six classes; see
data/README.md for the TACO -> six-class mapping).
Usage: python -m recyclebin.taco_eval --model mobilenetv2 --csv data/taco_spotcheck.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import tensorflow as tf

from .config import load_config
from .data import make_dataset
from .stats import wilson_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "mobilenetv2"], default="mobilenetv2")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes = cfg["classes"]
    df = pd.read_csv(args.csv).rename(columns={"true_label": "label"})
    model = tf.keras.models.load_model(Path(cfg["output"]["models_dir"]) / f"{args.model}.keras")
    probs = model.predict(make_dataset(df, cfg))
    df["pred"] = [classes[i] for i in probs.argmax(1)]
    df["confidence"] = probs.max(1)
    df["correct"] = df["pred"] == df["label"]
    k, n = int(df["correct"].sum()), len(df)
    lo, hi = wilson_ci(k, n)
    out = Path(cfg["output"]["results_dir"]) / "taco"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"taco_predictions_{args.model}.csv", index=False)
    print(df[["path", "label", "pred", "correct"]])
    print(f"\nTACO accuracy: {k}/{n} = {k/n:.1%}  (Wilson 95% CI {lo:.1%}-{hi:.1%})")
    print("Class mix of the spot-check:\n", df["label"].value_counts())


if __name__ == "__main__":
    main()
