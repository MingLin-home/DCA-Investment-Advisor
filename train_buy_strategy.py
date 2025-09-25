"""Entry point for training a dollar-cost-averaging buy strategy."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train buy strategy using settings from a YAML config file.",
    )
    parser.add_argument(
        "--config",
        default="./config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    config_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a top-level mapping of settings.")

    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    # The cfg variable is intentionally kept within main for further use.
    print(f"Loaded configuration with {len(cfg)} top-level keys from '{args.config}'.")


if __name__ == "__main__":
    main()
