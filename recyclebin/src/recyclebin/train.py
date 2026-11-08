"""Train a model.  Usage:  python -m recyclebin.train --model {baseline,mobilenetv2}"""
import argparse
from pathlib import Path
import pandas as pd
import tensorflow as tf

from .config import load_config
from .data import build_splits, make_dataset
from .models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "mobilenetv2"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tf.keras.utils.set_random_seed(cfg["seed"])
    df = build_splits(cfg)
    train = make_dataset(df[df.split == "train"], cfg, training=True)
    val = make_dataset(df[df.split == "val"], cfg)

    mc = cfg[args.model]
    model = build_model(args.model, cfg)
    model.compile(optimizer=tf.keras.optimizers.Adam(mc["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    models_dir = Path(cfg["output"]["models_dir"]); models_dir.mkdir(exist_ok=True)
    ckpt = models_dir / f"{args.model}.keras"
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=mc["early_stopping_patience"],
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
    ]
    hist = model.fit(train, validation_data=val, epochs=mc["epochs"], callbacks=cbs)
    out = Path(cfg["output"]["results_dir"]) / args.model; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist.history).to_csv(out / "training_history.csv", index=False)
    print("Saved", ckpt)


if __name__ == "__main__":
    main()
