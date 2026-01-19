from dataclasses import asdict
from typing import Optional

import numpy as np
from jax import numpy as jnp

from dipolesbi.tools.configs import (
    DataTransformConfig,
    DataTransformSpec,
    EmbeddingNetConfig,
    ThetaTransformConfig,
    ThetaTransformSpec,
    TransformConfig,
)
from dipolesbi.tools.hadamard_transform import HadamardTransform, HadamardTransformJax
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.transforms import DipoleBijectorWrapper, DipoleThetaTransform, ZScore


def _map_tree(tree, fn):
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _map_tree(v, fn) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        mapped = [_map_tree(v, fn) for v in tree]
        return type(tree)(mapped)
    return fn(tree)


def serialize_transform_config(transform_config: TransformConfig) -> dict:
    return {
        "data_spec": asdict(transform_config.data_transform_config.spec),
        "theta_spec": asdict(transform_config.theta_transform_config.spec),
    }


def deserialize_transform_config(
    serialized: dict,
    prior: Optional[DipolePriorJax],
) -> TransformConfig:
    data_spec_dict = dict(serialized["data_spec"])
    embedding_cfg = data_spec_dict.get("embedding_config")
    if isinstance(embedding_cfg, dict):
        data_spec_dict["embedding_config"] = EmbeddingNetConfig(**embedding_cfg)
    data_spec = DataTransformSpec(**data_spec_dict)
    theta_spec = ThetaTransformSpec(**serialized["theta_spec"])
    data_cfg = DataTransformConfig(data_spec)
    theta_cfg = ThetaTransformConfig(
        spec=theta_spec,
        prior=prior if theta_spec.kind != "blank" else None,
    )
    return TransformConfig(data_cfg, theta_cfg)


def serialize_transform_state(transform_config: TransformConfig) -> dict:
    data_transform = transform_config.data_transform_config.data_transform
    theta_transform = transform_config.theta_transform_config.theta_transform

    if data_transform is None:
        data_state = {"kind": "none"}
    elif isinstance(data_transform, ZScore):
        data_state = {
            "kind": "zscore",
            "mu": None if data_transform.mu is None else np.asarray(data_transform.mu),
            "sigma": None if data_transform.sigma is None else np.asarray(data_transform.sigma),
            "method": data_transform.method,
        }
    elif isinstance(data_transform, (HadamardTransform, HadamardTransformJax)):
        data_state = {
            "kind": "hadamard",
            "mu_at_level_post": _map_tree(data_transform.mu_at_level_post, np.asarray),
            "std_at_level_post": _map_tree(data_transform.std_at_level_post, np.asarray),
            "empty_norm_stats_flag": bool(data_transform.empty_norm_stats_flag),
        }
    else:
        raise ValueError(f"Unsupported data transform type: {type(data_transform)}")

    if theta_transform is None:
        theta_state = {"kind": "none"}
    elif isinstance(theta_transform, DipoleThetaTransform):
        theta_state = {
            "kind": "dipole_theta",
            "theta_mean": (
                None if theta_transform.theta_mean is None else np.asarray(theta_transform.theta_mean)
            ),
            "theta_std": (
                None if theta_transform.theta_std is None else np.asarray(theta_transform.theta_std)
            ),
        }
    elif isinstance(theta_transform, DipoleBijectorWrapper):
        theta_state = {"kind": "dipole_bijector"}
    else:
        raise ValueError(f"Unsupported theta transform type: {type(theta_transform)}")

    return {"data_state": data_state, "theta_state": theta_state}


def restore_transform_state(transform_config: TransformConfig, state: dict) -> None:
    data_transform = transform_config.data_transform_config.data_transform
    theta_transform = transform_config.theta_transform_config.theta_transform

    data_state = state.get("data_state", {})
    if data_transform is None:
        if data_state and data_state.get("kind") not in ("none", None):
            raise ValueError("Data transform state provided but no data transform exists.")
    elif isinstance(data_transform, ZScore):
        if data_state.get("kind") != "zscore":
            raise ValueError("Data transform state kind does not match ZScore.")
        data_transform.mu = (
            None if data_state.get("mu") is None else np.asarray(data_state["mu"])
        )
        data_transform.sigma = (
            None if data_state.get("sigma") is None else np.asarray(data_state["sigma"])
        )
    elif isinstance(data_transform, (HadamardTransform, HadamardTransformJax)):
        if data_state.get("kind") != "hadamard":
            raise ValueError("Data transform state kind does not match HadamardTransform.")
        xp = getattr(data_transform, "xp", np)
        data_transform.mu_at_level_post = _map_tree(
            data_state.get("mu_at_level_post"),
            lambda a: xp.asarray(a),
        )
        data_transform.std_at_level_post = _map_tree(
            data_state.get("std_at_level_post"),
            lambda a: xp.asarray(a),
        )
        data_transform.empty_norm_stats_flag = bool(
            data_state.get("empty_norm_stats_flag", True)
        )
    else:
        raise ValueError(f"Unsupported data transform type: {type(data_transform)}")

    theta_state = state.get("theta_state", {})
    if theta_transform is None:
        if theta_state and theta_state.get("kind") not in ("none", None):
            raise ValueError("Theta transform state provided but no theta transform exists.")
    elif isinstance(theta_transform, DipoleThetaTransform):
        if theta_state.get("kind") != "dipole_theta":
            raise ValueError("Theta transform state kind does not match DipoleThetaTransform.")
        theta_transform._theta_mean = (
            None if theta_state.get("theta_mean") is None else jnp.asarray(theta_state["theta_mean"])
        )
        theta_transform._theta_std = (
            None if theta_state.get("theta_std") is None else jnp.asarray(theta_state["theta_std"])
        )
    elif isinstance(theta_transform, DipoleBijectorWrapper):
        if theta_state.get("kind") not in ("dipole_bijector", None):
            raise ValueError("Theta transform state kind does not match DipoleBijectorWrapper.")
    else:
        raise ValueError(f"Unsupported theta transform type: {type(theta_transform)}")
