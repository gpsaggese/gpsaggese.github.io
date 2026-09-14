"""
Configuration management for ai_outreach campaigns.

Loads YAML config files that specify data sources, API keys, campaign
settings, and CRM paths.

Import as:

import ai_outreach.config.config as aoucocon
"""

import logging
import os
from typing import Any, Dict, Optional

import yaml  # type: ignore[import-untyped]

import helpers.hdbg as hdbg

_LOG = logging.getLogger(__name__)


# Default config template.
_DEFAULT_CONFIG = {
    "crm": {
        "db_path": "crm.db",
    },
    "api_keys": {
        "sendgrid": "SENDGRID_API_KEY",
        "hunterio": "HUNTERIO_API_KEY",
        "dropcontact": "DROPCONTACT_API_KEY",
        "phantombuster": "PHANTOMBUSTER_API_KEY",
    },
    "email": {
        "from_email": "",
        "from_name": "",
        "reply_to": "",
    },
    "data_sources": {},
}


def load_config(
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load campaign configuration from a YAML file.

    If no path is provided, returns the default config.

    :param config_path: path to YAML config file
    :return: configuration dictionary
    """
    if config_path is None:
        _LOG.info("No config file specified, using defaults")
        return _DEFAULT_CONFIG.copy()
    hdbg.dassert_path_exists(config_path)
    _LOG.info("Loading config from: %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Merge with defaults so missing keys get default values.
    merged = _DEFAULT_CONFIG.copy()
    _deep_merge(merged, config)
    return merged


def _deep_merge(base: Dict, override: Dict) -> None:
    """
    Recursively merge `override` into `base` in-place.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_api_key(config: Dict[str, Any], service: str) -> str:
    """
    Get an API key from environment variable named in config.

    :param config: configuration dictionary
    :param service: service name (e.g., "sendgrid", "hunterio")
    :return: API key string
    """
    hdbg.dassert_in(service, config["api_keys"])
    env_var = config["api_keys"][service]
    api_key = os.environ.get(env_var)
    hdbg.dassert_is_not(
        api_key,
        None,
        "Environment variable '%s' must be set for service '%s'",
        env_var,
        service,
    )
    return api_key


def get_default_config_template() -> str:
    """
    Return a YAML string of the default config template.

    Useful for generating a starter config file.
    """
    config_template = _DEFAULT_CONFIG.copy()
    config = str(
        yaml.dump(config_template, default_flow_style=False, sort_keys=False)
    )
    return config
