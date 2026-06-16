"""
config_loader.py
----------------
YAML configuration loader with path resolution and basic validation.
All other modules import configs through this single gateway so that
config access is consistent and errors surface early.
"""

import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Resolve project root as the directory that contains this utils/ package
# PROJECT/src/utils/config_loader.py  →  PROJECT/
# ---------------------------------------------------------------------------
_SRC_DIR     = Path(__file__).resolve().parent.parent   # PROJECT/src/
_PROJECT_DIR = _SRC_DIR.parent                          # PROJECT/


def get_project_root() -> Path:
    """Return the absolute path to the PROJECT root directory."""
    return _PROJECT_DIR


def _resolve_paths(cfg: dict, root: Path) -> dict:
    """
    Walk a config dict and convert every value under a 'paths' key from a
    relative string into an absolute Path, resolved against the project root.
    Nested dicts are handled recursively.
    """
    resolved = {}
    for key, value in cfg.items():
        if key == "paths" and isinstance(value, dict):
            resolved[key] = {
                k: root / v for k, v in value.items()
            }
        elif isinstance(value, dict):
            resolved[key] = _resolve_paths(value, root)
        else:
            resolved[key] = value
    return resolved


def load_config(config_name: str, resolve_paths: bool = True) -> dict:
    """
    Load a YAML config file from the PROJECT/configs/ directory.

    Parameters
    ----------
    config_name : str
        Filename without extension, e.g. 'dataset', 'model', 'training'.
    resolve_paths : bool
        If True, convert relative path strings to absolute Path objects.

    Returns
    -------
    dict
        Parsed and optionally path-resolved configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    yaml.YAMLError
        If the file cannot be parsed.
    """
    configs_dir = _PROJECT_DIR / "configs"
    config_path = configs_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs: {[f.stem for f in configs_dir.glob('*.yaml')]}"
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if resolve_paths:
        cfg = _resolve_paths(cfg, _PROJECT_DIR)

    return cfg


def load_all_configs() -> dict:
    """
    Convenience loader that returns a merged dict with keys
    'dataset', 'model', and 'training', each containing its YAML config.
    """
    return {
        "dataset":  load_config("dataset"),
        "model":    load_config("model"),
        "training": load_config("training"),
    }


def get_class_list(dataset_cfg: dict | None = None) -> list[str]:
    """
    Return the ordered list of all 46 ISL class names.
    Alphabets first (A-Z), then words.
    """
    if dataset_cfg is None:
        dataset_cfg = load_config("dataset", resolve_paths=False)
    alphabets = dataset_cfg["classes"]["alphabets"]
    words     = dataset_cfg["classes"]["words"]
    return alphabets + words


def get_num_classes(dataset_cfg: dict | None = None) -> int:
    """Return the total number of ISL classes (46)."""
    if dataset_cfg is None:
        dataset_cfg = load_config("dataset", resolve_paths=False)
    return int(dataset_cfg["num_classes"])


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist. Return path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
