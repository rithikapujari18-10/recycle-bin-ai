"""Stratified train/validation/test split and tf.data pipelines for TrashNet."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png"}


def build_splits(cfg, force=False):
    """Create (or load) a stratified split saved as CSV so every run uses the same files.

    NOTE: TrashNet has several photos of similar objects. A random split can place
    near-duplicates in different partitions (leakage). If object IDs are known,
    split by object/group instead and document it in the paper.
    """
    out = Path(cfg["data"]["splits_dir"]) / "splits.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    root = Path(cfg["data"]["trashnet_dir"])
    rows = []
    for cls in cfg["classes"]:
        for p in sorted((root / cls).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                rows.append((str(p), cls))
    df = pd.DataFrame(rows, columns=["path", "label"])
    seed = cfg["seed"]
    val, test = cfg["data"]["val_frac"], cfg["data"]["test_frac"]

    trainval, test_df = train_test_split(df, test_size=test, stratify=df["label"], random_state=seed)
    train_df, val_df = train_test_split(
        trainval, test_size=val / (1 - test), stratify=trainval["label"], random_state=seed)
    train_df, val_df, test_df = (d.assign(split=s) for d, s in
                                 ((train_df, "train"), (val_df, "val"), (test_df, "test")))
    full = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    print(full.groupby(["split", "label"]).size().unstack(fill_value=0))
    return full


def make_dataset(df, cfg, training=False):
    """Return a tf.data.Dataset of (image float32 in [0,255], integer label)."""
    import tensorflow as tf

    classes = cfg["classes"]
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    size = tuple(cfg["image"]["size"])
    paths = df["path"].tolist()
    labels = [cls_to_idx[l] for l in df["label"]]

    aug = None
    if training:
        a = cfg["augmentation"]
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(a["flip"]),
            tf.keras.layers.RandomRotation(a["rotation"]),
            tf.keras.layers.RandomZoom(a["zoom"]),
        ])

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, size)
        img.set_shape((*size, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=cfg["seed"])
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg["image"]["batch_size"])
    if aug is not None:
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
