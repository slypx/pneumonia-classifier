# utils/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Logging / checkpoints
LOG_DIR = PROJECT_ROOT / "logs"
TB_LOG_DIR = LOG_DIR / "tensorboard"
CHECKPOINT_PATH = LOG_DIR / "best_model.pth"

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
