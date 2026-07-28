import argparse
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from catsim import Racs, RacsConfig
from catsim.racs_jax import RacsJax
import numpy as np

from dipolesbi.pipelines.racs_observation_helpers import (
    build_mask_from_observation_config,
    build_real_sample,
    load_catalogue,
    load_observation_config,
)


def prepare_reference_observation(
    observation_config: Mapping[str, Any],
    catalogue_path: str | Path,
    paf_temperature_data_dir: str | Path,
    *,
    include_native: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Construct the configured RACS reference data vector and mask."""
    args = observation_config["args"]
    catalogue_path = Path(catalogue_path).expanduser().resolve(strict=True)
    paf_temperature_data_dir = (
        Path(paf_temperature_data_dir).expanduser().resolve(strict=True)
    )

    mask = build_mask_from_observation_config(observation_config)

    use_jax = bool(args["use_jax"])
    model_config = RacsConfig(
        product=args["racs_epoch"],
        catalogue_path=str(catalogue_path),
        flux_min=args["flux_min"],
        nside=args["nside"],
        chunk_size=args["chunk_size"],
        use_float32=False,
        downscale_nside=args["downscale_nside"],
        store_final_samples=not use_jax,
        fractional_error_flux_min_mjy=args["fractional_error_flux_min_mjy"],
        flux_temperature_min_mjy=args["flux_temperature_min_mjy"],
        paf_temperature_data_dir=str(paf_temperature_data_dir),
        temperature_fallback=(
            "open_meteo" if args["openmeteo_fallback"] else "none"
        ),
        mask_map=mask,
    )
    model = RacsJax(model_config) if use_jax else Racs(model_config)
    model.initialise_data()

    catalogue = load_catalogue(catalogue_path)
    prepared = build_real_sample(
        model,
        catalogue,
        args["flux_min"],
        list(args["summary_features"]),
        local_source_crossmatch_radius_arcsec=(
            args["local_source_crossmatch_radius_arcsec"]
        ),
        flux_temperature_min_mjy=args["flux_temperature_min_mjy"],
        flux_temperature_n_bins=args["flux_temperature_n_bins"],
        flux_temperature_quantiles=tuple(args["flux_temperature_quantiles"]),
        flux_elevation_n_bins=args["flux_elevation_n_bins"],
        flux_elevation_quantiles=tuple(args["flux_elevation_quantiles"]),
        save_map_plot=False,
    )
    if not include_native:
        return prepared

    original_downscale_nside = model.downscale_nside
    try:
        model.downscale_nside = None
        native = build_real_sample(
            model,
            catalogue,
            args["flux_min"],
            [],
            local_source_crossmatch_radius_arcsec=(
                args["local_source_crossmatch_radius_arcsec"]
            ),
            save_map_plot=False,
        )
    finally:
        model.downscale_nside = original_downscale_nside
    return *prepared, *native


def save_reference_observation(
    output_path: str | Path,
    x0: np.ndarray,
    mask: np.ndarray,
) -> Path:
    """Validate and atomically save a reference observation NPZ."""
    output = Path(output_path).expanduser()
    data = np.asarray(x0, dtype=np.float32)
    data_mask = np.asarray(mask, dtype=np.bool_)
    if data.ndim != 1 or data_mask.ndim != 1:
        raise ValueError("Reference data and mask must be one-dimensional.")
    if data.shape != data_mask.shape:
        raise ValueError(
            "Reference data and mask must have identical shapes; "
            f"got {data.shape} and {data_mask.shape}."
        )

    if output.is_file():
        try:
            with np.load(output, allow_pickle=False) as existing:
                unchanged = (
                    set(existing.files) == {"x0", "mask"}
                    and np.array_equal(existing["x0"], data, equal_nan=True)
                    and np.array_equal(existing["mask"], data_mask)
                )
        except (OSError, ValueError):
            unchanged = False
        if unchanged:
            return output

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.stem}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, x0=data, mask=data_mask)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def construct_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a reusable RACS reference observation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Observation YAML configuration.",
    )
    parser.add_argument(
        "--catalogue-path",
        type=Path,
        required=True,
        help="Resolved path to the catalogue selected by the workflow site config.",
    )
    parser.add_argument(
        "--paf-temperature-data-dir",
        type=Path,
        required=True,
        help="Resolved root of the PAF temperature collection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NPZ containing x0 and mask.",
    )
    parser.add_argument(
        "--native-output",
        type=Path,
        help="Optional native-resolution map-only NPZ containing x0 and mask.",
    )
    return parser


def main() -> None:
    cli_args = construct_argparser().parse_args()
    observation_config = load_observation_config(cli_args.config)
    prepared = prepare_reference_observation(
        observation_config,
        cli_args.catalogue_path,
        cli_args.paf_temperature_data_dir,
        include_native=cli_args.native_output is not None,
    )
    x0, mask = prepared[:2]
    output = save_reference_observation(cli_args.output, x0, mask)
    print(f"Saved reference observation: {output}")
    if cli_args.native_output is not None:
        native_x0, native_mask = prepared[2:]
        native_output = save_reference_observation(
            cli_args.native_output,
            native_x0,
            native_mask,
        )
        print(f"Saved native reference observation: {native_output}")


if __name__ == "__main__":
    main()
