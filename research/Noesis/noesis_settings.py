"""
Settings for the Noesis gateway, read once at startup.

Values come from `research/Noesis/.env` overlaid with the process environment
(the environment wins). Invalid values fail at startup, not on the first
request.

For the overall design see `research/Noesis/docs/gateway.README.md`.

Import as:

import research.Noesis.noesis_settings as rnonoset
"""

import dataclasses
import logging
import os
from typing import Dict, Mapping, Optional

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)

# #############################################################################
# Constants
# #############################################################################

# Directory containing this file, i.e., `research/Noesis`.
NOESIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATABASE_URL = "postgresql://noesis:noesis@localhost:5433/noesis_dev"
DEFAULT_MODEL_ID = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_SEED_PATH = os.path.join(NOESIS_DIR, "config", "seed.yaml")


# #############################################################################
# Settings
# #############################################################################


@dataclasses.dataclass(frozen=True)
class Settings:
    """
    Gateway configuration.

    E.g., with no overrides the gateway serves Llama 3.3 70B, caps completions
    at 256 tokens, and times a request out at 2x the contract latency limit.
    """

    # Needed only when the gateway uses the real OpenRouter provider; "" if
    # unset.
    openrouter_api_key: str
    database_url: str
    model_id: str
    # Task definition (PRD D1): completions are clamped to this many tokens.
    max_tokens_cap: int
    # Monthly spend cap on real provider cost (PRD G8); not enforced yet, see
    # `docs/mvp_decisions.md`.
    spend_cap_usd: float
    # Per-request timeout = `timeout_multiplier` * contract `l_max` (PRD G6).
    timeout_multiplier: float
    seed_path: str


# #############################################################################
# Loading
# #############################################################################


def read_dotenv(path: str) -> Dict[str, str]:
    """
    Parse simple `KEY=value` lines from a dotenv file.

    Blank lines and `#` comments are skipped; surrounding quotes are removed.

    :param path: dotenv file; a missing file yields an empty dict
    :return: parsed key/value pairs
    """
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get_positive(env: Mapping[str, str], name: str, default: str) -> float:
    """
    Read `name` from `env` as a positive number.

    :return: the parsed value
    """
    raw = env.get(name, default)
    hdbg.dassert(
        raw.replace(".", "", 1).isdigit(),
        "Setting '%s' must be a positive number, got '%s'",
        name,
        raw,
    )
    value = float(raw)
    hdbg.dassert_lt(0, value, "Setting '%s' must be > 0", name)
    return value


def load_settings(
    *,
    environ: Optional[Mapping[str, str]] = None,
    dotenv_path: str = "",
) -> Settings:
    """
    Build `Settings` from the dotenv file overlaid with the environment.

    :param environ: variables to read
        - Default: `os.environ`
    :param dotenv_path: dotenv file to read first
        - Default: `research/Noesis/.env`
    :return: validated settings
    """
    dotenv_path = dotenv_path or os.path.join(NOESIS_DIR, ".env")
    _LOG.debug("dotenv_path='%s'", dotenv_path)
    env = dict(read_dotenv(dotenv_path))
    env.update(os.environ if environ is None else environ)
    max_tokens_cap = _get_positive(env, "NOESIS_MAX_TOKENS_CAP", "256")
    hdbg.dassert(
        max_tokens_cap.is_integer(), "NOESIS_MAX_TOKENS_CAP must be an integer"
    )
    settings = Settings(
        openrouter_api_key=env.get("OPENROUTER_API_KEY", ""),
        database_url=env.get("NOESIS_DATABASE_URL", DEFAULT_DATABASE_URL),
        model_id=env.get("NOESIS_MODEL_ID", DEFAULT_MODEL_ID),
        max_tokens_cap=int(max_tokens_cap),
        spend_cap_usd=_get_positive(env, "NOESIS_SPEND_CAP_USD", "50"),
        timeout_multiplier=_get_positive(env, "NOESIS_TIMEOUT_MULTIPLIER", "2.0"),
        seed_path=env.get("NOESIS_SEED_PATH", DEFAULT_SEED_PATH),
    )
    return settings
