from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SOURCE_SPLITS = ("train", "val", "test")
SOURCE_CLASSES = {"benign": 0, "malign": 1}
PATIENT_RE = re.compile(r"(P_\d+)", re.IGNORECASE)


def find_dataset_root(path: Path) -> Path:
    """Accept either the real dataset root or its one-folder outer wrapper."""
    path = path.expanduser().resolve()

    def is_root(candidate: Path) -> bool:
        return all(
            (candidate / split / class_name).is_dir()
            for split in SOURCE_SPLITS
            for class_name in SOURCE_CLASSES
        )

    if is_root(path):
        return path

    if path.is_dir():
        matches = [child for child in path.iterdir() if child.is_dir() and is_root(child)]
        if len(matches) == 1:
            return matches[0].resolve()

    expected = "train, val and test folders, each containing benign and malign"
    raise SystemExit(f"ERROR: {path} is not the revised CBIS-DDSM root. Expected {expected}.")


def patient_from_name(path: Path) -> str:
    match = PATIENT_RE.search(path.stem)
    return match.group(1).upper() if match else ""


def laterality_from_name(path: Path) -> str:
    name = path.stem.upper()
    if "_LEFT_" in name:
        return "LEFT"
    if "_RIGHT_" in name:
        return "RIGHT"
    return ""


def view_from_name(path: Path) -> str:
    name = path.stem.upper()
    if name.endswith("_MLO") or "_MLO_" in name:
        return "MLO"
    if name.endswith("_CC") or "_CC_" in name:
        return "CC"
    return ""


def abnormality_from_name(path: Path) -> str:
    name = path.stem.lower()
    if name.startswith("calc"):
        return "calcification"
    if name.startswith("mass"):
        return "mass"
    return "unknown"


def scan_source(source: Path) -> list[dict]:
    records: list[dict] = []
    missing_patient: list[str] = []

    for source_split in SOURCE_SPLITS:
        for source_class, label in SOURCE_CLASSES.items():
            folder = source / source_split / source_class
            for image_path in sorted(folder.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                patient = patient_from_name(image_path)
                if not patient:
                    missing_patient.append(str(image_path.relative_to(source)))
                    continue
                records.append(
                    {
                        "source_relative_path": image_path.relative_to(source).as_posix(),
                        "source_split": source_split,
                        "source_class": source_class,
                        "label": label,
                        "patient_id": patient,
                        "laterality": laterality_from_name(image_path),
                        "view": view_from_name(image_path),
                        "abnormality_type": abnormality_from_name(image_path),
                    }
                )

    if missing_patient:
        examples = "\n".join(f"  - {name}" for name in missing_patient[:5])
        raise SystemExit(
            "ERROR: patient IDs could not be read from some filenames. "
            f"Patient-safe splitting cannot continue. Examples:\n{examples}"
        )
    if not records:
        raise SystemExit("ERROR: no supported mammogram images were found in the dataset folders.")
    return records


def choose_validation_patients(records: list[dict], fraction: float, seed: int) -> set[str]:
    patient_label: dict[str, int] = defaultdict(int)
    for record in records:
        if record["source_split"] != "test":
            patient_label[record["patient_id"]] = max(
                patient_label[record["patient_id"]], record["label"]
            )

    strata: dict[int, list[str]] = {0: [], 1: []}
    for patient, label in patient_label.items():
        strata[label].append(patient)

    rng = random.Random(seed)
    validation: set[str] = set()
    for label in (0, 1):
        patients = sorted(strata[label])
        rng.shuffle(patients)
        count = max(1, round(len(patients) * fraction)) if len(patients) > 1 else 0
        count = min(count, max(0, len(patients) - 1))
        validation.update(patients[:count])
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register the revised public CBIS-DDSM PNG dataset without copying it."
    )
    parser.add_argument("source", type=Path, help="Extracted cbis_ddsm_rev folder")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.05 <= args.val_fraction <= 0.30:
        raise SystemExit("ERROR: --val-fraction must be between 0.05 and 0.30.")

    source = find_dataset_root(args.source)
    records = scan_source(source)

    original_train_patients = {
        record["patient_id"] for record in records if record["source_split"] != "test"
    }
    original_test_patients = {
        record["patient_id"] for record in records if record["source_split"] == "test"
    }
    source_overlap = original_train_patients & original_test_patients
    if source_overlap:
        examples = ", ".join(sorted(source_overlap)[:10])
        raise SystemExit(
            "ERROR: patient overlap exists between the source train/val and test sets: "
            f"{examples}. Use the revised cbis_ddsm_rev dataset, not the original split."
        )

    validation_patients = choose_validation_patients(records, args.val_fraction, args.seed)
    for record in records:
        if record["source_split"] == "test":
            record["split"] = "test"
        elif record["patient_id"] in validation_patients:
            record["split"] = "validation"
        else:
            record["split"] = "train"
        record["class_name"] = "malignant" if record["label"] else "benign"

    split_patients: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        split_patients[record["split"]].add(record["patient_id"])
        counts[record["split"]][record["class_name"]] += 1

    overlap = {
        "train_validation": len(split_patients["train"] & split_patients["validation"]),
        "train_test": len(split_patients["train"] & split_patients["test"]),
        "validation_test": len(split_patients["validation"] & split_patients["test"]),
    }
    if any(overlap.values()):
        raise SystemExit(f"ERROR: patient leakage detected after splitting: {overlap}")

    project = Path(__file__).resolve().parents[1]
    output = project / "data" / "cbis_ddsm_prepared"
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"ERROR: output exists: {output}. Use --overwrite to rebuild it.")
    output.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_relative_path",
        "split",
        "class_name",
        "label",
        "patient_id",
        "laterality",
        "view",
        "abnormality_type",
        "source_split",
        "source_class",
    ]
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    config = {
        "dataset": "dpetrini/cbis_ddsm_rev",
        "source_root": str(source),
        "manifest": "manifest.csv",
        "images_are_copied": False,
        "class_mapping": {"benign": 0, "malign": 1},
    }
    (output / "dataset_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    source_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        source_counts[record["source_split"]][record["source_class"]] += 1

    summary = {
        "dataset": "dpetrini/cbis_ddsm_rev",
        "source_root": str(source),
        "seed": args.seed,
        "validation_fraction_of_source_train_and_val_patients": args.val_fraction,
        "source_images": {
            split: {
                "benign": source_counts[split]["benign"],
                "malign": source_counts[split]["malign"],
                "total": sum(source_counts[split].values()),
            }
            for split in SOURCE_SPLITS
        },
        "final_patients": {
            split: len(split_patients[split])
            for split in ("train", "validation", "test")
        },
        "final_images": {
            split: {
                "benign": counts[split]["benign"],
                "malignant": counts[split]["malignant"],
                "total": sum(counts[split].values()),
            }
            for split in ("train", "validation", "test")
        },
        "patient_overlap": overlap,
        "images_copied_into_project": False,
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nDataset registered from: {source}")
    print(f"Manifest saved to: {output / 'manifest.csv'}")
    print("STEP 1 COMPLETE. All patient-overlap counts are 0.")
    print("Keep the extracted dataset in this location while training and evaluating.")


if __name__ == "__main__":
    main()
