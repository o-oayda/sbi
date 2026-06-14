import argparse
from types import MethodType
from typing import Literal
from catsim import RACS_PRODUCTS, Racs, RacsConfig, RacsJax
from catsim.utils.healsphere import downgrade_ignore_nan
from dipoleutils.utils.samples import CatalogueToMap
import healpy as hp
import jax
import numpy as np
from dipolesbi.tools.configs import DataTransformSpec, Scenario
from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.ui import MultiRoundInfererUI
from dipolesbi.tools.utils import batch_simulate
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.mask import Masker
import matplotlib.pyplot as plt

NativeCountSummary = Literal["hist_logprob", "hist_ilr", "log_dispersion"]
NativeCountHistTransform = Literal["logprob", "ilr"]

# FREE_TEMP_PIVOT_MODEL = "free_temp_pivot"
# FIXED_TEMP_PIVOT_25C_MODEL = "fixed_temp_pivot_25c"
# MODEL_CHOICES = [FREE_TEMP_PIVOT_MODEL, FIXED_TEMP_PIVOT_25C_MODEL]


def _parse_modes(raw_modes: list[str] | None, parser: argparse.ArgumentParser) -> list[str]:
    modes: list[str] = []
    for entry in raw_modes or []:
        modes.extend(part.strip().upper() for part in entry.split(",") if part.strip())
    if not modes:
        parser.error("At least one mode must be provided via --mode.")
    return modes


def _model_product(model: Racs | RacsJax):
    product = getattr(model, "product", None)
    if product is not None:
        return product
    return model.cfg.product


def _build_real_sample(
    model: Racs | RacsJax,
    flux_min: float,
    append_native_count_summary: bool = False,
    hist_max_count: int = 30,
    hist_eps: float = 1e-6,
    native_count_summary: NativeCountSummary = "hist_logprob",
) -> tuple[np.ndarray, np.ndarray]:
    product = _model_product(model)
    cat = DataLoader(*product.data_loader_args).load()
    c2map = CatalogueToMap(cat)
    c2map.make_cut(product.columns.total_flux, minimum=flux_min, maximum=None)
    if "Name" in cat.colnames:
        c2map.crossmatch_local_sources(
            'equatorial',
            radius=5,
            source_name_A_column='Name',
        )
    density_map = c2map.make_density_map(
        coordinate_system='equatorial',
        nside=model.nside,
        nest=True
    ).astype('float32')

    mask = model.mask_map.astype(np.bool_, copy=False)
    density_map = density_map.copy()
    density_map[~mask] = np.nan

    hp.projview(density_map, nest=True)
    plt.savefig(f'racs_{product.key}.png')
    plt.close()

    if append_native_count_summary:
        if model.downscale_nside is None:
            raise ValueError(
                "--append_native_count_summary requires --downscale_nside so the "
                "map part of the hybrid target is well-defined."
            )
        return _build_hybrid_sample_from_native(
            density_map,
            mask,
            downscale_nside=model.downscale_nside,
            hist_max_count=hist_max_count,
            hist_eps=hist_eps,
            native_count_summary=native_count_summary,
        )

    if model.downscale_nside is None:
        return density_map, mask

    coarse_map, coarse_mask = downgrade_ignore_nan(
        density_map,
        mask,
        model.downscale_nside,
    )
    coarse_map = coarse_map.astype(np.float32, copy=False)
    coarse_mask = coarse_mask.astype(np.bool_, copy=False)
    coarse_map = coarse_map.copy()
    coarse_map[~coarse_mask] = np.nan
    return coarse_map, coarse_mask


def _helmert_ilr_basis(n_bins: int) -> np.ndarray:
    if n_bins < 2:
        raise ValueError("ILR basis requires at least two bins.")

    basis = np.zeros((n_bins, n_bins - 1), dtype=np.float64)
    for col in range(n_bins - 1):
        scale = np.sqrt((col + 1) * (col + 2))
        basis[: col + 1, col] = 1.0 / scale
        basis[col + 1, col] = -(col + 1) / scale
    return basis


def _native_counts(
    native_map: np.ndarray,
    native_mask: np.ndarray,
) -> np.ndarray:
    if native_map.shape != native_mask.shape:
        raise ValueError("native_map and native_mask must have matching shapes.")

    valid = native_mask.astype(bool, copy=False) & np.isfinite(native_map)
    if not np.any(valid):
        raise ValueError("Cannot build native-count summary with no unmasked native pixels.")

    counts = np.asarray(native_map[valid])
    rounded_counts = np.rint(counts)
    if np.any(rounded_counts < 0):
        raise ValueError("Native-count summary received negative counts.")
    return rounded_counts.astype(np.int64)


def _native_count_hist_probabilities(
    native_map: np.ndarray,
    native_mask: np.ndarray,
    max_count: int,
    eps: float,
) -> np.ndarray:
    if max_count < 1:
        raise ValueError("hist_max_count must be at least 1.")
    if eps < 0:
        raise ValueError("hist_eps must be non-negative.")

    # implements hist ceiling, k >= max_count
    bin_index = np.minimum(_native_counts(native_map, native_mask), max_count)
    hist = np.bincount(bin_index, minlength=max_count + 1)[: max_count + 1]
    smoothed_hist = hist.astype(np.float64) + eps
    probabilities = smoothed_hist / float(smoothed_hist.sum())
    return probabilities


def _native_count_log_dispersion_feature(
    native_map: np.ndarray,
    native_mask: np.ndarray,
) -> np.ndarray:
    counts = _native_counts(native_map, native_mask).astype(np.float64)
    if counts.size < 2:
        raise ValueError("Native-count log dispersion requires at least two pixels.")
    mean = float(np.mean(counts))
    if mean <= 0.0:
        raise ValueError("Native-count log dispersion requires positive mean counts.")
    dispersion = float(np.var(counts, ddof=1) / mean)
    if dispersion <= 0.0:
        raise ValueError("Native-count log dispersion requires positive dispersion.")
    return np.asarray([np.log(dispersion)], dtype=np.float32)


def _native_count_summary_features(
    native_map: np.ndarray,
    native_mask: np.ndarray,
    max_count: int,
    eps: float,
    summary: NativeCountSummary = "hist_logprob",
) -> np.ndarray:
    if eps <= 0:
        raise ValueError("hist_eps must be positive.")
    if summary == "hist_logprob":
        probabilities = _native_count_hist_probabilities(
            native_map,
            native_mask,
            max_count=max_count,
            eps=0.0,
        )
        return np.log(probabilities + eps).astype(np.float32)
    if summary == "hist_ilr":
        # ILR is the summary-coordinate definition for this experiment, not a
        # runtime InvertibleDataTransform. The learned likelihood is in
        # coarse_map + ILR(histogram) coordinates, with no Jacobian tracked
        # back to raw normalized histogram probabilities. This coordinate
        # choice is shared by model comparisons and is theta-independent.
        probabilities = _native_count_hist_probabilities(
            native_map,
            native_mask,
            max_count=max_count,
            eps=eps,
        )
        basis = _helmert_ilr_basis(max_count + 1)
        features = np.log(probabilities) @ basis
        return features.astype(np.float32)
    if summary == "log_dispersion":
        return _native_count_log_dispersion_feature(native_map, native_mask)

    raise ValueError(f"Unknown native-count summary: {summary}")


def _native_count_hist_features(
    native_map: np.ndarray,
    native_mask: np.ndarray,
    max_count: int,
    eps: float,
    transform: NativeCountHistTransform = "logprob",
) -> np.ndarray:
    summary = _summary_from_legacy_hist_transform(transform)
    return _native_count_summary_features(
        native_map,
        native_mask,
        max_count=max_count,
        eps=eps,
        summary=summary,
    )


def _inverse_ilr_to_probabilities(z: np.ndarray, basis: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("basis must be a two-dimensional array.")
    if z.shape[-1] != basis.shape[1]:
        raise ValueError("ILR coordinate dimension does not match basis.")

    clr = z @ basis.T
    clr = clr - np.max(clr, axis=-1, keepdims=True)
    probabilities = np.exp(clr)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True)


def _native_count_summary_ndim(
    hist_max_count: int,
    summary: NativeCountSummary,
) -> int:
    if summary == "hist_logprob":
        return hist_max_count + 1
    if summary == "hist_ilr":
        return hist_max_count
    if summary == "log_dispersion":
        return 1
    raise ValueError(f"Unknown native-count summary: {summary}")


def _native_count_hist_feature_ndim(
    hist_max_count: int,
    transform: NativeCountHistTransform,
) -> int:
    return _native_count_summary_ndim(
        hist_max_count,
        _summary_from_legacy_hist_transform(transform),
    )


def _summary_from_legacy_hist_transform(
    transform: NativeCountHistTransform,
) -> NativeCountSummary:
    if transform == "logprob":
        return "hist_logprob"
    if transform == "ilr":
        return "hist_ilr"
    raise ValueError(f"Unknown native count histogram transform: {transform}")


def _build_hybrid_sample_from_native(
    native_map: np.ndarray,
    native_mask: np.ndarray,
    downscale_nside: int,
    hist_max_count: int,
    hist_eps: float,
    native_count_summary: NativeCountSummary = "hist_logprob",
    native_count_hist_transform: NativeCountHistTransform | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if native_count_hist_transform is not None:
        native_count_summary = _summary_from_legacy_hist_transform(
            native_count_hist_transform
        )
    coarse_map, coarse_mask = downgrade_ignore_nan(
        native_map,
        native_mask,
        downscale_nside,
    )
    coarse_map = coarse_map.astype(np.float32, copy=False)
    coarse_mask = coarse_mask.astype(np.bool_, copy=False)
    coarse_map = coarse_map.copy()
    coarse_map[~coarse_mask] = np.nan

    summary_features = _native_count_summary_features(
        native_map,
        native_mask,
        max_count=hist_max_count,
        eps=hist_eps,
        summary=native_count_summary,
    )
    summary_mask = np.ones(summary_features.shape, dtype=np.bool_)

    hybrid = np.concatenate([coarse_map, summary_features]).astype(np.float32, copy=False)
    hybrid_mask = np.concatenate([coarse_mask, summary_mask]).astype(np.bool_, copy=False)
    return hybrid, hybrid_mask

def _build_mask(nside: int) -> np.ndarray:
    masker = Masker(np.ones(hp.nside2npix(nside)), coordinate_system='equatorial')

    masker.mask_galactic_plane(5)
    masker.mask_a_team_sources(radius_deg=2)
    masker.mask_equatorial_poles(north_radius=42)

    masker.mask_a_team_sources(radius_deg=3, source_names=['Cygnus A'])
    masker.mask_a_team_sources(radius_deg=13, source_names=['LMC'])
    masker.mask_a_team_sources(radius_deg=8, source_names=['SMC'])

    maskmap = masker.get_mask_map()
    return hp.reorder(maskmap, r2n=True)


def build_racs_config(
    *,
    racs_epoch: str,
    flux_min: float,
    nside: int,
    chunk_size: int,
    use_jax: bool,
    cluster_count_model: str,
    downscale_nside: int | None,
    alpha_mean: float,
    alpha_sigma: float,
    fractional_error_flux_min_mjy: float,
    mask_map: np.ndarray,
    max_cluster_children_per_parent: int,
    openmeteo_fallback: bool = False,
) -> RacsConfig:
    return RacsConfig(
        product=racs_epoch,
        flux_min=flux_min,
        nside=nside,
        chunk_size=chunk_size,
        use_float32=False,
        cluster_count_model=cluster_count_model,
        downscale_nside=downscale_nside,
        store_final_samples=not use_jax,
        alpha_mean=alpha_mean,
        alpha_sigma=alpha_sigma,
        fractional_error_flux_min_mjy=fractional_error_flux_min_mjy,
        paf_temperature_data_dir='/home/oliver/Documents/dipole-utils/data/paf_temps',
        temperature_fallback="open_meteo" if openmeteo_fallback else "none",
        mask_map=mask_map,
        max_cluster_children_per_parent=max_cluster_children_per_parent,
    )


def build_prior_and_reference_theta(
    model: Racs,
    # chosen_model: str = FREE_TEMP_PIVOT_MODEL,
    simulate_clustering: None | Literal['geometric', 'poisson'] = None,
) -> tuple[DipolePriorNP, dict[str, float]]:
    prior = DipolePriorNP(
        mean_count_range=[5.6, 6.8],
        speed_range=[0, 8],
    )
    prior.change_kwarg(
        param_short_name="N",
        new_kwarg="log10_n_initial_samples",
    )
    prior.add_prior(
        short_name='beta',
        simulator_kwarg='temp_beta',
        low=0.,
        high=0.05,
        dist_type='Uniform'
    )
    if simulate_clustering:
        if simulate_clustering == 'geometric':
            prior.add_prior(
                'pclus',
                simulator_kwarg='p_clus',
                low=0,
                high=1,
                dist_type='Uniform'
            )
            prior.add_prior(
                'pstop',
                simulator_kwarg='clus_stop_prob',
                low=0.4,
                high=1,
                dist_type='Uniform'
            )
        elif simulate_clustering == 'poisson':
            prior.add_prior(
                'lclus',
                simulator_kwarg='lambda_clus',
                low=0,
                high=3,
                dist_type='Uniform'
            )
        else:
            raise ValueError(f'{simulate_clustering} not recognised.')

    # prior.add_prior(
    #     short_name='eta',
    #     simulator_kwarg='fractional_error_eta',
    #     low=0.,
    #     high=200.,
    #     dist_type='Uniform'
    # )

    temp_beta_theta0 = 0.02
    # if chosen_model == FREE_TEMP_PIVOT_MODEL:
    #     prior.add_prior(
    #         short_name='T0',
    #         simulator_kwarg='temp_pivot_c',
    #         low=10,
    #         high=40,
    #         dist_type='Uniform'
    #     )
    #     temp_pivot_theta0 = np.nanmin(model.temperature_map) + (
    #         np.nanmax(model.temperature_map) - np.nanmin(model.temperature_map)
    #     ) / 2
    # elif chosen_model == FIXED_TEMP_PIVOT_25C_MODEL:
    #     temp_pivot_theta0 = 25.0
    # else:
    #     raise ValueError(f"Unknown RACS-low3 model: {chosen_model}")

    theta_0 = {
        "log10_n_initial_samples": 6.65,
        "observer_speed": 1.0,
        "dipole_longitude": 264.021,
        "dipole_latitude": 48.253,
        "temp_beta": temp_beta_theta0,
        # "temp_pivot_c": temp_pivot_theta0,
        # "temp_intercept": -temp_beta_theta0 + 1,
        "p_clus": 0.,
        "clus_stop_prob": 1.
        # "fractional_error_eta": 20.
    }
    if simulate_clustering == 'poisson':
        theta_0['lambda_clus'] = 0.

    return prior, theta_0#, temp_pivot_theta0


def make_simulator_wrapper(
    model: Racs,
    # chosen_model: str = FREE_TEMP_PIVOT_MODEL,
    native_output: bool = False,
    append_native_count_summary: bool = False,
    hist_max_count: int = 30,
    hist_eps: float = 1e-6,
    native_count_summary: NativeCountSummary = "hist_logprob",
):
    def simulator_wrapper(
        rng_key: NPKey | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        # temp_slope = kwargs['temp_slope']
        # kwargs['temp_intercept'] = -temp_slope + 1
        # if chosen_model == FIXED_TEMP_PIVOT_25C_MODEL:
        #     kwargs['temp_pivot_c'] = 25.0
        # elif chosen_model != FREE_TEMP_PIVOT_MODEL:
        #     raise ValueError(f"Unknown RACS-low3 model: {chosen_model}")
        if append_native_count_summary:
            if model.downscale_nside is None:
                raise ValueError(
                    "append_native_count_summary requires model.downscale_nside."
                )
            native_map, native_mask = _generate_dipole_native(
                model,
                rng_key=rng_key,
                **kwargs,
            )
            return _build_hybrid_sample_from_native(
                native_map,
                native_mask,
                downscale_nside=model.downscale_nside,
                hist_max_count=hist_max_count,
                hist_eps=hist_eps,
                native_count_summary=native_count_summary,
            )

        if native_output:
            return _generate_dipole_native(model, rng_key=rng_key, **kwargs)
        return model.generate_dipole(rng_key=rng_key, **kwargs)

    return simulator_wrapper


def _generate_dipole_native(
    model: Racs,
    *args,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    original_downscale_nside = model.downscale_nside
    try:
        model.downscale_nside = None
        return model.generate_dipole(*args, **kwargs)
    finally:
        model.downscale_nside = original_downscale_nside


# Local convenience only. Do not rely on this for joblib worker processes.
def attach_native_generate_dipole(model: Racs) -> None:
    if hasattr(model, "generate_dipole_native"):
        return

    def generate_dipole_native(self: Racs, *args, **kwargs):
        return _generate_dipole_native(self, *args, **kwargs)

    model.generate_dipole_native = MethodType(generate_dipole_native, model)  # type: ignore[attr-defined]


def build_scenario(
    mode: str,
    effective_nside: int,
    prior: DipolePriorNP,
    theta_0: dict[str, float],
    out_dir: str,
    ssnle_seed: int,
    n_rounds: int,
    n_simulations: int,
    map_ndim: int | None = None,
    summary_ndim: int | None = None,
    hist_max_count: int | None = None,
    hist_eps: float | None = None,
    native_count_summary: NativeCountSummary | None = None,
) -> Scenario:
    prior_jax = prior.to_jax()

    if mode == "NPE":
        return Scenario.anynside_npe(
            nside=effective_nside,
            theta_prior=prior_jax,
            reference_theta=theta_0,
            theta_spec_overrides={"embed_transform_in_flow": True},
            multiround_overrides={
                "prng_integer_seed": ssnle_seed,
                "plot_save_dir": out_dir,
                "n_rounds": n_rounds,
                "simulation_budget": n_simulations,
                "likelihood_chunk_size_gb": 0.5,
                "n_likelihood_samples": 10_000,
            },
            training_overrides={"learning_rate": 0.001},
        )

    if mode == "NLE":
        data_spec = DataTransformSpec.zscore(method="batchwise")
        return Scenario.anynside_nle(
            nside=effective_nside,
            theta_prior=prior_jax,
            training_overrides={
                "learning_rate": 1e-4,
                "min_lr_ratio": 1.0,
            },
            reference_theta=theta_0,
            multiround_overrides={
                "prng_integer_seed": ssnle_seed,
                "plot_save_dir": out_dir,
                "simulation_budget": n_simulations,
                "n_rounds": n_rounds,
                "likelihood_chunk_size_gb": 0.5,
                "n_likelihood_samples": 10_000,
                "map_ndim": map_ndim,
                "summary_ndim": summary_ndim,
                "native_count_hist_max_count": hist_max_count,
                "native_count_hist_eps": hist_eps,
                "native_count_summary": native_count_summary,
            },
            flow_overrides={
                "decoder_n_neurons": 128,
                "decoder_n_layers": 4,
                "architecture": 4 * ["MAF"] + ["surjective_MAF"] + 6 * ["MAF"],
                "data_reduction_factor": 0.5,
            },
            data_spec=data_spec,
        )

    raise KeyError(f"Mode {mode} not recognised.")


def make_model_sim_wrapper(
    simulator_wrapper,
    n_workers: int | None,
):
    def model_sim_wrapper(
        npkey: NPKey,
        params: dict[str, np.ndarray],
        noise: bool = True,
        ui: MultiRoundInfererUI | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return batch_simulate(
            params,
            simulator_wrapper,
            n_workers=n_workers,
            ui=ui,
            rng_key=npkey,
        )

    return model_sim_wrapper


def _jax_key_from_npkey(key: NPKey) -> jax.Array:
    if isinstance(key, NPKey):
        key_data = key._ss.generate_state(2, dtype=np.uint32)
    else:
        key_data = np.asarray(key, dtype=np.uint32).reshape(2)
    return jax.device_put(key_data)


def _batch_generate_dipole_native(
    model: RacsJax,
    theta: dict[str, np.ndarray],
    key: jax.Array,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    original_downscale_nside = model.downscale_nside
    try:
        model.downscale_nside = None
        return model.batch_generate_dipole(
            theta, key, batch_size=batch_size, show_progress=True
        )
    finally:
        model.downscale_nside = original_downscale_nside


def _build_hybrid_batch_from_native(
    native_maps: np.ndarray,
    native_masks: np.ndarray,
    downscale_nside: int,
    hist_max_count: int,
    hist_eps: float,
    native_count_summary: NativeCountSummary = "hist_logprob",
) -> tuple[np.ndarray, np.ndarray]:
    outputs = [
        _build_hybrid_sample_from_native(
            native_map,
            native_mask,
            downscale_nside=downscale_nside,
            hist_max_count=hist_max_count,
            hist_eps=hist_eps,
            native_count_summary=native_count_summary,
        )
        for native_map, native_mask in zip(native_maps, native_masks)
    ]
    x = np.stack([output[0] for output in outputs], axis=0)
    mask = np.stack([output[1] for output in outputs], axis=0)
    return x, mask


def make_jax_model_sim_wrapper(
    model: RacsJax,
    batch_size: int,
    append_native_count_summary: bool = False,
    hist_max_count: int = 30,
    hist_eps: float = 1e-6,
    native_count_summary: NativeCountSummary = "hist_logprob",
):
    def model_sim_wrapper(
        npkey: NPKey,
        params: dict[str, np.ndarray],
        noise: bool = True,
        ui: MultiRoundInfererUI | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        jax_key = _jax_key_from_npkey(npkey)
        theta = {key: np.asarray(value) for key, value in params.items()}

        if append_native_count_summary:
            if model.downscale_nside is None:
                raise ValueError(
                    "append_native_count_summary requires model.downscale_nside."
                )
            downscale_nside = model.downscale_nside
            native_maps, native_masks = _batch_generate_dipole_native(
                model,
                theta,
                jax_key,
                batch_size=batch_size,
            )
            return _build_hybrid_batch_from_native(
                native_maps,
                native_masks,
                downscale_nside=downscale_nside,
                hist_max_count=hist_max_count,
                hist_eps=hist_eps,
                native_count_summary=native_count_summary,
            )

        return model.batch_generate_dipole(
            theta, jax_key, batch_size=batch_size, show_progress=True
        )

    return model_sim_wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        help='One or more modes to run, separated by spaces (e.g. "--mode NLE NPE").',
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        required=True,
        help="Number of simulations to run.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=15,
        help="Number of rounds of inference.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of local workers to use for NumPy simulation batches.",
    )
    parser.add_argument(
        "--use_jax",
        action="store_true",
        help="Use catsim's JAX batched RACS simulator.",
    )
    parser.add_argument(
        "--jax_batch_size",
        type=int,
        default=5,
        help="Batch size passed to RacsJax.batch_generate_dipole.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save outputs into.",
    )
    # parser.add_argument(
    #     "--model",
    #     choices=MODEL_CHOICES,
    #     default=FREE_TEMP_PIVOT_MODEL,
    #     help=(
    #         "Choose whether temp_pivot_c is inferred freely or fixed by the simulator."
    #     ),
    # )
    parser.add_argument(
        "--ssnle_seed",
        type=int,
        default=0,
        help="Seed used for sequential neural estimators.",
    )
    parser.add_argument(
        "--downscale_nside",
        type=int,
        default=None,
        help="Optional HEALPix nside to downscale maps to.",
    )
    parser.add_argument(
        "--no_ui",
        action="store_true",
        help="Disable the Rich multi-round progress UI.",
    )
    parser.add_argument(
        "--flux_min",
        type=float,
        default=2.0,
        help="RACS flux threshold in mJy.",
    )
    parser.add_argument(
        "--racs_epoch",
        choices=sorted(RACS_PRODUCTS),
        default="low3",
        help="RACS product/epoch to simulate.",
    )
    parser.add_argument(
        "--openmeteo_fallback",
        action="store_true",
        help="Use Open-Meteo ambient temperatures when PAF temperatures are unavailable.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=64,
        help="Native HEALPix nside used by the simulator.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help=(
            "Chunk size used inside the simulator. Defaults to 2_500_000 for "
            "NumPy and 140_000 for JAX."
        ),
    )
    parser.add_argument(
        "--alpha_mean",
        type=float,
        default=0.8,
        help="Mean of the Gaussian spectral-index model.",
    )
    parser.add_argument(
        "--alpha_sigma",
        type=float,
        default=0.2,
        help="Standard deviation of the Gaussian spectral-index model.",
    )
    parser.add_argument(
        "--fractional_error_flux_min_mjy",
        type=float,
        default=10.0,
        help="Minimum flux used when building the empirical fractional-error lookup.",
    )
    parser.add_argument(
        "--simulate_clustering",
        type=str,
        help="Use clustering model to represent extended sources."
    )
    parser.add_argument(
        "--append_native_count_hist",
        action="store_true",
        help=(
            "Deprecated alias for --append_native_count_summary with "
            "--native_count_summary hist_logprob/hist_ilr."
        ),
    )
    parser.add_argument(
        "--append_native_count_summary",
        action="store_true",
        help=(
            "Append native-nside one-point count summary features to the "
            "downscaled map for NLE."
        ),
    )
    parser.add_argument(
        "--hist_max_count",
        type=int,
        default=30,
        help=(
            "First native count assigned to the overflow bin; explicit bins are "
            "0..hist_max_count-1 and overflow is >= hist_max_count."
        ),
    )
    parser.add_argument(
        "--hist_eps",
        type=float,
        default=1e-6,
        help=(
            "Positive histogram stabilizer: added inside log(p + eps) for "
            "logprob, or as an additive bin-count pseudocount for ilr."
        ),
    )
    parser.add_argument(
        "--native_count_hist_transform",
        choices=["logprob", "ilr"],
        default=None,
        help=(
            "Deprecated alias for --native_count_summary hist_logprob/hist_ilr."
        ),
    )
    parser.add_argument(
        "--native_count_summary",
        choices=["hist_logprob", "hist_ilr", "log_dispersion"],
        default=None,
        help=(
            "Native-count summary appended to the downscaled map. hist_logprob "
            "uses per-bin log(p + eps); hist_ilr uses additive hist_eps "
            "smoothing then Helmert ILR coordinates; log_dispersion appends "
            "log(var(counts) / mean(counts))."
        ),
    )
    parser.add_argument(
        "--max_children",
        type=int,
        default=16
    )
    args = parser.parse_args()

    if args.chunk_size is None:
        args.chunk_size = 140_000 if args.use_jax else 2_500_000
    if args.use_jax and args.n_workers is not None:
        parser.error("--n_workers is only used by the NumPy simulator.")
    if args.use_jax and args.jax_batch_size <= 0:
        parser.error("--jax_batch_size must be positive.")

    if (
        args.native_count_summary is not None
        and args.native_count_hist_transform is not None
    ):
        parser.error(
            "Use either --native_count_summary or --native_count_hist_transform, not both."
        )

    append_native_count_summary = (
        args.append_native_count_summary or args.append_native_count_hist
    )
    if args.native_count_summary is not None:
        native_count_summary = args.native_count_summary
    elif args.native_count_hist_transform is not None:
        native_count_summary = _summary_from_legacy_hist_transform(
            args.native_count_hist_transform
        )
    else:
        native_count_summary = "hist_logprob"

    if append_native_count_summary and args.downscale_nside is None:
        parser.error("--append_native_count_summary requires --downscale_nside.")

    modes = _parse_modes(args.mode, parser)
    if append_native_count_summary and any(mode != "NLE" for mode in modes):
        parser.error("--append_native_count_summary is currently supported for NLE only.")
    mask = _build_mask(args.nside)

    if not args.simulate_clustering:
        clus_model_cfg = 'geometric'
    else:
        clus_model_cfg = args.simulate_clustering

    config = build_racs_config(
        racs_epoch=args.racs_epoch,
        flux_min=args.flux_min,
        nside=args.nside,
        chunk_size=args.chunk_size,
        use_jax=args.use_jax,
        cluster_count_model=clus_model_cfg,
        downscale_nside=args.downscale_nside,
        alpha_mean=args.alpha_mean,
        alpha_sigma=args.alpha_sigma,
        fractional_error_flux_min_mjy=args.fractional_error_flux_min_mjy,
        openmeteo_fallback=args.openmeteo_fallback,
        mask_map=mask,
        max_cluster_children_per_parent=args.max_children,
    )
    model = RacsJax(config) if args.use_jax else Racs(config)
    model.initialise_data()

    x0, mask = _build_real_sample(
        model,
        flux_min=args.flux_min,
        append_native_count_summary=append_native_count_summary,
        hist_max_count=args.hist_max_count,
        hist_eps=args.hist_eps,
        native_count_summary=native_count_summary,
    )
    if append_native_count_summary:
        assert args.downscale_nside is not None
        map_ndim = hp.nside2npix(args.downscale_nside)
        summary_ndim = _native_count_summary_ndim(
            args.hist_max_count,
            native_count_summary,
        )
        effective_nside = args.downscale_nside
    else:
        map_ndim = None
        summary_ndim = None
        effective_nside = hp.npix2nside(x0.size)


    observed_map = x0[:map_ndim] if map_ndim is not None else x0
    observed_count = float(np.nansum(observed_map))
    if observed_count <= 0:
        product = _model_product(model)
        raise ValueError(
            f"Observed {product.label} map has zero total counts after masking/cuts."
        )

    prior, theta_0 = build_prior_and_reference_theta(
        model,
        # chosen_model=args.model,
        simulate_clustering=args.simulate_clustering
    )
    if args.use_jax:
        model_sim_wrapper = make_jax_model_sim_wrapper(
            model,
            batch_size=args.jax_batch_size,
            append_native_count_summary=append_native_count_summary,
            hist_max_count=args.hist_max_count,
            hist_eps=args.hist_eps,
            native_count_summary=native_count_summary,
        )
    else:
        simulator_wrapper = make_simulator_wrapper(
            model,
            # chosen_model=args.model,
            append_native_count_summary=append_native_count_summary,
            hist_max_count=args.hist_max_count,
            hist_eps=args.hist_eps,
            native_count_summary=native_count_summary,
        )
        model_sim_wrapper = make_model_sim_wrapper(
            simulator_wrapper=simulator_wrapper,
            n_workers=args.n_workers,
        )

    for mode in modes:
        scenario = build_scenario(
            mode=mode,
            effective_nside=effective_nside,
            prior=prior,
            theta_0=theta_0,
            out_dir=args.out_dir,
            ssnle_seed=args.ssnle_seed,
            n_rounds=args.n_rounds,
            n_simulations=args.n_simulations,
            map_ndim=map_ndim,
            summary_ndim=summary_ndim,
            hist_max_count=args.hist_max_count if append_native_count_summary else None,
            hist_eps=args.hist_eps if append_native_count_summary else None,
            native_count_summary=(
                native_count_summary if append_native_count_summary else None
            ),
        )
        inferer = MultiRoundInferer(
            mode,
            prior,
            model_sim_wrapper,
            (x0, mask),
            multi_round_config=scenario.multiround,
            transform_config=scenario.transforms,
            nflow_config=scenario.flow,
            train_config=scenario.training,
            use_ui=not args.no_ui,
            model_config=config
        )
        inferer.run()
