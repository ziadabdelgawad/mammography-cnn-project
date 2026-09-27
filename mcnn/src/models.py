from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import IMAGE_SIZE

INPUT_SHAPE = (*IMAGE_SIZE, 3)


def _augmentation():
    return keras.Sequential(
        [
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def build_baseline():
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="baseline_cnn")
    return _compile(model, 1e-3)


def build_regularized():
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = _augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="regularized_cnn")
    return _compile(model, 5e-4)


def build_transfer():
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = _augmentation()(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
    )
    # Transfer learning + limited fine-tuning: only VGG16 block5 is trainable.
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5")
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="vgg16_transfer")
    return _compile(model, 1e-5)


def _compile(model, learning_rate: float):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_model(name: str):
    builders = {
        "baseline": build_baseline,
        "regularized": build_regularized,
        "transfer": build_transfer,
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}. Choose from {sorted(builders)}")
    return builders[name]()
