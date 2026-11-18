"""Central configuration. Values are read from environment variables (and `.env`) at call time,
so tests can override paths with monkeypatch without re-importing modules."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SEED: int = 42
ALERT_BUDGET: float = 0.01  # top 1% of transactions become alerts
SPLIT_FRACTIONS: tuple[float, float, float] = (0.6, 0.2, 0.2)


@dataclass(frozen=True)
class Settings:
    """Filesystem locations used by the pipeline and the API."""

    data_path: Path
    patterns_path: Path
    model_dir: Path
    processed_dir: Path
    db_path: Path
    frontend_dir: Path


def get_settings() -> Settings:
    """Build settings from environment variables, falling back to repo-relative defaults."""
    return Settings(
        data_path=Path(os.getenv("AML_DATA_PATH", ROOT / "data" / "raw" / "SAML-D.csv")),
        patterns_path=Path(os.getenv("AML_PATTERNS_PATH", ROOT / "data" / "raw" / "HI-Small_Patterns.txt")),
        model_dir=Path(os.getenv("AML_MODEL_DIR", ROOT / "models")),
        processed_dir=Path(os.getenv("AML_PROCESSED_DIR", ROOT / "data" / "processed")),
        db_path=Path(os.getenv("AML_DB_PATH", ROOT / "data" / "aml.db")),
        frontend_dir=ROOT / "frontend",
    )
