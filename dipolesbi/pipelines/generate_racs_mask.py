import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from dipolesbi.pipelines.racs_observation_helpers import (
    build_mask_from_observation_config,
    load_observation_config,
)


def save_mask(output_path: str | Path, mask: np.ndarray) -> Path:
    """Validate and atomically save a boolean HEALPix mask."""
    output = Path(output_path).expanduser()
    data = np.asarray(mask)
    if data.dtype != np.bool_:
        raise ValueError(f"Mask must have boolean dtype; got {data.dtype}.")
    if data.ndim != 1:
        raise ValueError("Mask must be one-dimensional.")

    if output.is_file():
        try:
            unchanged = np.array_equal(
                np.load(output, allow_pickle=False),
                data,
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
            suffix=".npy",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.save(temporary, data)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def save_mask_plot(output_path: str | Path, mask: np.ndarray) -> Path:
    """Save an unannotated native-resolution projection of a mask."""
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    hp.projview(mask, nest=True)
    plt.savefig(output)
    plt.close()
    return output


def construct_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the HEALPix mask for a RACS observation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Observation YAML configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NPY containing the boolean mask.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        required=True,
        help="Output image containing an unannotated HEALPix projection.",
    )
    return parser


def main() -> None:
    cli_args = construct_argparser().parse_args()
    observation_config = load_observation_config(cli_args.config)
    mask = build_mask_from_observation_config(observation_config)
    expected_size = hp.nside2npix(observation_config["args"]["nside"])
    if mask.shape != (expected_size,):
        raise ValueError(
            f"Generated mask must contain {expected_size} pixels; got {mask.shape}."
        )
    output = save_mask(cli_args.output, mask)
    plot_output = save_mask_plot(cli_args.plot_output, mask)
    print(f"Saved mask: {output}")
    print(f"Saved mask projection: {plot_output}")


if __name__ == "__main__":
    main()
