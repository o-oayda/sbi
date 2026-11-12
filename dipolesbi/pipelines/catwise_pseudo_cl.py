"""Native-resolution coupled count spectra for the CatWISE summary."""

import healpy as hp
import numpy as np


def native_pseudo_cl(counts, mask, observation):
    """Return unbinned masked C_ell, including shot noise, in the chosen range."""
    try:
        import pymaster as nmt
    except ImportError as error:
        raise ImportError(
            'The native pseudo-Cl summary requires pymaster; '
            'install with uv sync --extra glass.'
        ) from error

    counts = np.asarray(counts, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not np.all(np.isfinite(counts[mask])) or np.any(counts[mask] < 0):
        raise ValueError(
            'Pseudo-Cl requires finite nonnegative unmasked counts.'
        )
    mean = counts[mask].mean()
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError('Pseudo-Cl requires a positive unmasked mean count.')
    # Zero outside the mask before transforming; native masked pixels are NaN.
    delta = np.zeros_like(counts)
    delta[mask] = counts[mask] / mean - 1.0
    lmin, lmax = observation['pseudo_cl_range']
    field = nmt.NmtField(
        hp.reorder(mask.astype(float), n2r=True),
        [hp.reorder(delta, n2r=True)],
        lmax=lmax,
    )
    spectrum = np.asarray(nmt.compute_coupled_cell(field, field))[0]
    result = spectrum[lmin : lmax + 1]
    if result.shape != (lmax - lmin + 1,) or not np.all(np.isfinite(result)):
        raise ValueError('NaMaster returned an invalid pseudo-Cl spectrum.')
    return result
