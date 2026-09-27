from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import tensorflow as tf

from .config import BATCH_SIZE, DATASET_DIR, IMAGE_SIZE, SEED


def _dataset_metadata() -> tuple[Path, list[dict]]:
    config_path = DATASET_DIR / "dataset_config.json"
    manifest_path = DATASET_DIR / "manifest.csv"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Prepared dataset metadata was not found. Run STEP1_PREPARE_DATASET.bat first."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_root = os.environ.get("CBIS_DDSM_ROOT", config.get("source_root", ""))
    source_root = Path(configured_root).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"The registered dataset folder no longer exists: {source_root}\n"
            "Move it back, set CBIS_DDSM_ROOT, or run STEP1_PREPARE_DATASET.bat again."
        )

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Dataset manifest is empty: {manifest_path}")
    return source_root, rows


def _rows_for_split(split: str) -> tuple[list[str], list[float]]:
    source_root, rows = _dataset_metadata()
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise RuntimeError(f"No images were assigned to the {split!r} split.")

    paths = [str(source_root / row["source_relative_path"]) for row in selected]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} dataset images are missing. Examples:\n{examples}\n"
            "Run STEP1_PREPARE_DATASET.bat again if the dataset was moved."
        )
    labels = [float(row["label"]) for row in selected]
    return paths, labels


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    encoded = tf.io.read_file(path)
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    image = tf.cast(image, tf.float32)
    label = tf.reshape(tf.cast(label, tf.float32), [1])
    return image, label


def _make_dataset(split: str, *, shuffle: bool) -> tf.data.Dataset:
    paths, labels = _rows_for_split(split)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        _decode_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_datasets() -> dict[str, tf.data.Dataset]:
    return {
        "train": _make_dataset("train", shuffle=True),
        "validation": _make_dataset("validation", shuffle=False),
        "test": _make_dataset("test", shuffle=False),
    }


def class_weights_from_manifest() -> dict[int, float]:
    _, rows = _dataset_metadata()
    counts = [
        sum(1 for row in rows if row["split"] == "train" and int(row["label"]) == label)
        for label in (0, 1)
    ]
    if min(counts, default=0) == 0:
        raise RuntimeError(f"Training classes are empty or missing: {counts}")
    total = sum(counts)
    return {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1]),
    }
