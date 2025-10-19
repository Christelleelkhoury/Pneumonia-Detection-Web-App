from pathlib import Path

ROOT            = Path(__file__).parent
CHECKPOINT_PATH = ROOT / "resources"
NUM_CLASSES = 2
LABEL_MAP   = {1: "Pneumonia"}
IMAGE_SIZE  = (1024, 1024) # H, W
