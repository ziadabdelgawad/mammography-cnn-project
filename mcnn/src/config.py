from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "cbis_ddsm_prepared"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
CLASS_NAMES = ["benign", "malignant"]
