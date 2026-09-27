from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import ARTIFACTS_DIR, MODELS_DIR, SEED
from .data import class_weights_from_manifest, load_datasets
from .models import build_model


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_history(history, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.history.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        n = len(next(iter(history.history.values())))
        for i in range(n):
            writer.writerow([i + 1, *[history.history[k][i] for k in keys]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["baseline", "regularized", "transfer"])
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = ARTIFACTS_DIR / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets()
    class_weights = class_weights_from_manifest()
    model = build_model(args.model)
    model.summary()

    model_path = MODELS_DIR / f"{args.model}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        datasets["train"],
        validation_data=datasets["validation"],
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    save_history(history, run_dir / "training_history.csv")

    best_val_auc = max(history.history.get("val_auc", [float("nan")]))
    metadata = {
        "model": args.model,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_auc_during_training": float(best_val_auc),
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "model_file": str(model_path),
    }
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Best model saved to: {model_path}")


if __name__ == "__main__":
    main()
