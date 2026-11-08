"""Baseline CNN (from scratch) and MobileNetV2 (ImageNet initialisation)."""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(cfg):
    """Simple CNN trained from scratch.

    CONFIRM: replace with the exact baseline architecture from the team notebook.
    """
    h, w = cfg["image"]["size"]
    n = len(cfg["classes"])
    return models.Sequential([
        layers.Input((h, w, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(n, activation="softmax"),
    ], name="baseline_cnn")


def build_mobilenetv2(cfg):
    """MobileNetV2 backbone -> global average pooling -> dense 6-way softmax."""
    h, w = cfg["image"]["size"]
    n = len(cfg["classes"])
    c = cfg["mobilenetv2"]
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(h, w, 3))
    base.trainable = not c["freeze_backbone"]

    inputs = layers.Input((h, w, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # [0,255] -> [-1,1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    if c.get("dropout", 0) > 0:
        x = layers.Dropout(c["dropout"])(x)
    outputs = layers.Dense(n, activation="softmax")(x)
    return models.Model(inputs, outputs, name="mobilenetv2_transfer")


def build_model(name, cfg):
    return {"baseline": build_baseline_cnn, "mobilenetv2": build_mobilenetv2}[name](cfg)
