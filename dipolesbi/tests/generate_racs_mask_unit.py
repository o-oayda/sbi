import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml

from dipolesbi.lib.yaml_to_mask import yaml_to_mask
from dipolesbi.pipelines import generate_racs_mask as generation
from dipolesbi.pipelines import racs_observation_helpers as helpers


def test_yaml_to_mask_builds_nested_equatorial_cut(tmp_path):
    config_path = tmp_path / "mask.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "coordinates": "ANY",
                "nside": "ANY",
                "ordering": "ANY",
                "cuts": ["dec>0"],
            }
        ),
        encoding="utf-8",
    )

    mask = yaml_to_mask(
        config_path,
        coordinates="equatorial",
        nside=8,
        ordering="NESTED",
    )

    assert mask.dtype == np.bool_
    assert mask.shape == (12 * 8**2,)
    assert 0 < np.count_nonzero(mask) < mask.size


def test_yaml_to_mask_rejects_conflicting_pinned_pixelisation(tmp_path):
    config_path = tmp_path / "mask.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "coordinates": "equatorial",
                "nside": 4,
                "ordering": "NESTED",
                "pixels": [0],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contradicts the YAML file"):
        yaml_to_mask(config_path, nside=8)


def test_config_and_additional_masks_are_combined_by_intersection(monkeypatch):
    configured = np.array([True, True, False, False])
    additional_ring = np.array([True, False, True, False])
    calls = []

    monkeypatch.setattr(
        helpers,
        "yaml_to_mask",
        lambda path, **kwargs: calls.append((path, kwargs)) or configured,
    )

    class FakeMasker:
        def __init__(self, values, coordinate_system):
            assert coordinate_system == "equatorial"

        def mask_galactic_plane(self, width):
            calls.append(("galactic", width))

        def get_mask_map(self):
            return additional_ring

    monkeypatch.setattr(helpers.hp, "nside2npix", lambda nside: 4)
    monkeypatch.setattr(helpers.hp, "reorder", lambda mask, r2n: mask)
    monkeypatch.setattr(helpers, "Masker", FakeMasker)
    observation = {
        "args": {"nside": 8},
        "mask": {"config": "mask.yaml", "galactic_plane_width_deg": 5},
    }

    result = helpers.build_mask_from_observation_config(observation)

    np.testing.assert_array_equal(result, configured & additional_ring)
    assert calls[0] == (
        "mask.yaml",
        {"coordinates": "equatorial", "nside": 8, "ordering": "NESTED"},
    )
    assert calls[1] == ("galactic", 5)


def test_config_only_mask_does_not_apply_legacy_defaults(monkeypatch):
    configured = np.array([True, False, True, False])

    class FakeMasker:
        def __init__(self, values, coordinate_system):
            pass

        def __getattr__(self, name):
            raise AssertionError(f"unexpected Masker operation: {name}")

    monkeypatch.setattr(helpers.hp, "nside2npix", lambda nside: 4)
    monkeypatch.setattr(helpers, "Masker", FakeMasker)
    monkeypatch.setattr(helpers, "yaml_to_mask", lambda *args, **kwargs: configured)

    result = helpers.build_mask_from_observation_config(
        {"args": {"nside": 8}, "mask": {"config": "mask.yaml"}}
    )

    np.testing.assert_array_equal(result, configured)


def test_save_mask_round_trip_and_preserves_unchanged_mtime(tmp_path):
    path = tmp_path / "nested" / "mask.npy"
    mask = np.array([True, False, True])

    assert generation.save_mask(path, mask) == path
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), mask)
    old_mtime_ns = 1_000_000_000
    os.utime(path, ns=(old_mtime_ns, old_mtime_ns))

    generation.save_mask(path, mask)

    assert path.stat().st_mtime_ns == old_mtime_ns


def test_save_mask_plot_uses_plain_nested_projview(tmp_path, monkeypatch):
    calls = []
    mask = np.array([True, False])
    monkeypatch.setattr(
        generation.hp,
        "projview",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(generation.plt, "savefig", lambda path: calls.append(path))
    monkeypatch.setattr(generation.plt, "close", lambda: None)
    output = tmp_path / "mask.png"

    generation.save_mask_plot(output, mask)

    assert calls == [((mask,), {"nest": True}), output]


@pytest.mark.parametrize(
    "mask",
    [np.ones((2, 2), dtype=bool), np.ones(2, dtype=np.int8)],
)
def test_save_mask_rejects_invalid_arrays(tmp_path, mask):
    with pytest.raises(ValueError, match="boolean dtype|one-dimensional"):
        generation.save_mask(tmp_path / "mask.npy", mask)


@pytest.mark.parametrize(
    "mask_config",
    [
        {"config": "workflow/configs/observations/masks/racs_mid1_mask.yaml"},
        {
            "galactic_plane_width_deg": 5,
            "north_equatorial_pole_radius_deg": 42,
            "default_a_team_radius_deg": 2,
        },
        {"config": "mask.yaml", "source_radii_deg": {"LMC": 13}},
    ],
)
def test_observation_schema_accepts_supported_mask_forms(mask_config):
    schema_path = Path("workflow/schemas/racs-observation.schema.yaml")
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))

    jsonschema.validate(mask_config, schema["properties"]["mask"])


def test_observation_schema_rejects_empty_mask():
    schema_path = Path("workflow/schemas/racs-observation.schema.yaml")
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({}, schema["properties"]["mask"])
