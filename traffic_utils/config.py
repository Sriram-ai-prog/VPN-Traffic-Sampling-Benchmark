# config.py
import os
import random
import numpy as np
import torch
import warnings

# --- Global Reproducibility ---
SEED: int = 42

def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(SEED)

# --- Dataset Config ---
TARGET_COL: str = "traffic_type"
CATEGORICAL_COLS = [
    "is_continuous_flow", "no_forward_packets",
    "no_backward_packets", "single_packet_flow",
]

# --- Device ---
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# --- Experiment Constants ---
OPTUNA_TRIALS: int = 20
CV_FOLDS: int = 5
WGAN_EPOCHS: int = 50
CTGAN_EPOCHS: int = 30
MAX_CVR_THRESHOLD: float = 0.05

# --- Environment Safety ---
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "14")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)