"""
config.py
---------
Shared project config loader (same pattern as the da-id study; no code dependency).
"""

import logging
from pathlib import Path

import yaml

# project root (parent of src/)
ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Load settings.yaml as a dict."""
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    """Stdout logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
