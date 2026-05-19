from __future__ import annotations

import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


MODEL_CONFIG_SCHEMA_VERSION = 1


def _is_dataclass_instance(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_ready(item())
        except ValueError:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def save_model_config(config: object, path: str | Path) -> None:
    """Save a dataclass model config as a portable JSON sidecar."""
    if not _is_dataclass_instance(config):
        raise TypeError("model config must be a dataclass instance.")

    config_type = type(config)
    payload = {
        "schema_version": MODEL_CONFIG_SCHEMA_VERSION,
        "class_module": config_type.__module__,
        "class_name": config_type.__name__,
        "fields": _json_ready(asdict(config)),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_model_config(path: str | Path) -> object:
    """Load a model config previously written by :func:`save_model_config`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    schema_version = payload.get("schema_version")
    if schema_version != MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model config schema version {schema_version!r}."
        )

    class_module = payload.get("class_module")
    class_name = payload.get("class_name")
    fields = payload.get("fields")
    if not isinstance(class_module, str) or not class_module:
        raise ValueError("model config JSON is missing a valid class_module.")
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("model config JSON is missing a valid class_name.")
    if not isinstance(fields, dict):
        raise ValueError("model config JSON is missing a valid fields mapping.")

    module = importlib.import_module(class_module)
    config_type = getattr(module, class_name)
    if not isinstance(config_type, type) or not is_dataclass(config_type):
        raise TypeError(
            f"{class_module}.{class_name} is not a dataclass config type."
        )
    return config_type(**fields)
