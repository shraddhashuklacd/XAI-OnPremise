import os
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: str = str(BASE_DIR)
OUTPUT_DIR: str = str(BASE_DIR / "outputs")
LOG_DIR: str = str(BASE_DIR / "logs")

for _dir in (OUTPUT_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)
