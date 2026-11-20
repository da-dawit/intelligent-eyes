"""Where data lives. Data stays outside the repo; set EYES_DATA (and EYES_RESULTS) to move it."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("EYES_DATA", REPO.parent / "data"))
RESULTS = Path(os.environ.get("EYES_RESULTS", REPO / "results"))


def processed(dataset: str) -> Path:
    return DATA / "processed" / dataset


def features(dataset: str, encoder: str) -> Path:
    return DATA / "features" / dataset / encoder
