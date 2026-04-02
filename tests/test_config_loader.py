"""配置加载器测试。"""

from __future__ import annotations

import yaml

from src.config_loader import load_config


def _write_yaml(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def test_load_config_merges_local_override(tmp_path):
    config_path = tmp_path / "config.yaml"
    local_path = tmp_path / "config.local.yaml"
    _write_yaml(config_path, {"models": {"embedding": {"device": "cpu", "batch_size": 32}}})
    _write_yaml(local_path, {"models": {"embedding": {"batch_size": 64}}})

    payload = load_config(config_path=config_path)

    assert payload["models"]["embedding"]["device"] == "cpu"
    assert payload["models"]["embedding"]["batch_size"] == 64


def test_env_main_config_overrides_default_path(monkeypatch, tmp_path):
    env_config_path = tmp_path / "env-config.yaml"
    explicit_config_path = tmp_path / "explicit-config.yaml"
    _write_yaml(env_config_path, {"env_marker": "from_env"})
    _write_yaml(explicit_config_path, {"env_marker": "from_explicit"})
    monkeypatch.setenv("NEV_CONFIG_PATH", str(env_config_path))

    payload_default = load_config()
    payload_explicit = load_config(config_path=explicit_config_path)

    assert payload_default["env_marker"] == "from_env"
    assert payload_explicit["env_marker"] == "from_explicit"


def test_env_local_config_is_used_when_local_not_explicit(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.yaml"
    env_local_path = tmp_path / "custom.local.from.env.yaml"
    explicit_local_path = tmp_path / "custom.local.explicit.yaml"

    _write_yaml(config_path, {"a": 1, "nested": {"b": 1}})
    _write_yaml(env_local_path, {"a": 2})
    _write_yaml(explicit_local_path, {"nested": {"b": 3}})

    monkeypatch.setenv("NEV_LOCAL_CONFIG_PATH", str(env_local_path))

    payload_env_local = load_config(config_path=config_path)
    payload_explicit_local = load_config(
        config_path=config_path,
        local_config_path=explicit_local_path,
    )

    assert payload_env_local["a"] == 2
    assert payload_env_local["nested"]["b"] == 1
    assert payload_explicit_local["a"] == 1
    assert payload_explicit_local["nested"]["b"] == 3
