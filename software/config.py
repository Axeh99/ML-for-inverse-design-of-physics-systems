from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import yaml


@dataclass(frozen=True)
class ProjectConfig:
    target_columns: List[str]
    feature_columns: List[str]
    base_features: Dict[str, Any]
    search_space: Dict[str, Any]
    requirements: Dict[str, Any]
    cols: List[str]


def load_config(
    config_path: str = "config.yaml",
    fallback_path: Optional[str] = None,
) -> ProjectConfig:
    p = Path(config_path)
    if not p.exists():
        if fallback_path is None:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        p = Path(fallback_path)
        if not p.exists():
            raise FileNotFoundError(f"Fallback config file not found: {fallback_path}")

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    required_keys = ["target_columns", "feature_columns", "base_features", "search_space", "requirements", "cols"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in config: {missing}")

    return ProjectConfig(
        target_columns=list(data["target_columns"]),
        feature_columns=list(data["feature_columns"]),
        base_features=dict(data["base_features"]),
        search_space=dict(data["search_space"]),
        requirements=dict(data["requirements"]),
        cols=list(data["cols"]),
    )
