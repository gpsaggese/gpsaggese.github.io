"""Paths, the pinned corpus and runtime settings for the offline half."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".cache"
CLONE_DIR = CACHE_DIR / "connector"
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
SPECS_DIR = ROOT / "specs"
GOLD_DIR = SPECS_DIR / "gold"

REPO_URL = "https://github.com/eclipse-edc/Connector.git"
SHA = "a66db8963acfecec5a7f3316115f42eaf54c0374"
RECORD_ROOT = "docs/developer/decision-records/"
RECORDS = [
    "2025-05-28-dataspace-profile-context",
    "2025-07-25-multiple-participant-identifiers",
    "2025-08-26-participant-context",
    "2025-10-15-participant-context-config",
    "2025-10-29-participant-identifiers",
    "2026-05-11-multi-profile-virtual-connector",
    "2026-09-22-per-participant-resource-ids",
    "2026-09-24-remove-atomikos",
    "2025-02-07-http-proxy-data-plane-deprecation",
    "2024-10-10-daps-deprecation",
]

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """The model settings are optional. Without them the build uses the rule-based parser only."""
    openai_api_key: str | None
    chat_model: str | None
    stale_warn_threshold: float


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        chat_model=os.environ.get("OPENAI_CHAT_MODEL") or None,
        stale_warn_threshold=float(os.environ.get("STALE_WARN_THRESHOLD", 15)),
    )
