"""
统一配置加载。

规则：
- 默认读取 config/config.yaml
- 若存在同目录的 config.local.yaml，则递归覆盖
"""

from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并配置字典。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if local_config_path is None:
        local_config_path = config_path.with_name(
            f"{config_path.stem}.local{config_path.suffix}"
        )
    else:
        local_config_path = Path(local_config_path)

    if Path(local_config_path).exists():
        with Path(local_config_path).open(encoding="utf-8") as f:
            local_config = yaml.safe_load(f) or {}
        config = deep_merge(config, local_config)

    return config
