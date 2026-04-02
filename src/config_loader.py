"""
统一配置加载。

规则：
- 默认读取 config/config.yaml
- 若存在同目录的 config.local.yaml，则递归覆盖
- 当使用默认主配置路径时，支持通过环境变量 NEV_CONFIG_PATH 覆盖
- 支持通过环境变量 NEV_LOCAL_CONFIG_PATH 覆盖本地配置路径
"""

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
CONFIG_PATH_ENV = "NEV_CONFIG_PATH"
LOCAL_CONFIG_PATH_ENV = "NEV_LOCAL_CONFIG_PATH"


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并配置字典。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _is_default_config_path(config_path: Path) -> bool:
    return config_path == DEFAULT_CONFIG_PATH or config_path.resolve() == DEFAULT_CONFIG_PATH.resolve()


def _resolve_main_config_path(config_path: Path) -> Path:
    env_path = os.getenv(CONFIG_PATH_ENV, "").strip()
    if env_path and _is_default_config_path(config_path):
        return Path(env_path)
    return config_path


def _resolve_local_config_path(config_path: Path, local_config_path: str | Path | None) -> Path:
    if local_config_path is not None:
        return Path(local_config_path)

    env_local_path = os.getenv(LOCAL_CONFIG_PATH_ENV, "").strip()
    if env_local_path:
        return Path(env_local_path)

    return config_path.with_name(f"{config_path.stem}.local{config_path.suffix}")


def load_config(
    config_path: str | Path = "config/config.yaml",
    local_config_path: str | Path | None = None,
) -> dict:
    """
    加载主配置，并按需叠加本地覆盖配置。

    Args:
        config_path: 主配置文件路径
        local_config_path: 本地覆盖文件路径。默认推导为同目录下的 config.local.yaml
    """
    config_path = _resolve_main_config_path(Path(config_path))
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    local_config_path = _resolve_local_config_path(config_path, local_config_path)

    if Path(local_config_path).exists():
        with Path(local_config_path).open(encoding="utf-8") as f:
            local_config = yaml.safe_load(f) or {}
        config = deep_merge(config, local_config)

    return config
