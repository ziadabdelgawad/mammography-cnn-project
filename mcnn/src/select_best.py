from __future__ import annotations

import json
import shutil

from .config import ARTIFACTS_DIR, MODELS_DIR


def main():
    candidates = []
    for name in ("baseline", "regularized", "transfer"):
        metrics_path = ARTIFACTS_DIR / name / "metrics.json"
        model_path = MODELS_DIR / f"{name}.keras"
        if metrics_path.is_file() and model_path.is_file():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            candidates.append((float(data["validation"]["roc_auc"]), name, data, model_path))
    if not candidates:
        raise SystemExit("No evaluated models found. Run EVALUATE_ALL.bat first.")

    # IMPORTANT: final deployment model is selected using validation performance only.
    candidates.sort(reverse=True, key=lambda x: x[0])
    val_auc, name, data, source = candidates[0]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "best_model.keras"
    shutil.copy2(source, target)
    metadata = {
        "selected_model": name,
        "selection_metric": "validation_roc_auc",
        "validation_roc_auc": val_auc,
        "threshold": data["threshold_selected_on_validation"],
        "source_model": str(source.name),
        "note": "Model selected without using test-set performance.",
    }
    (MODELS_DIR / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected: {name} (validation ROC-AUC={val_auc:.4f})")
    print(f"Deployment model: {target}")


if __name__ == "__main__":
    main()
