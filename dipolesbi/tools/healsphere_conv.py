# Adapted from https://github.com/aasensio/sphericalCNN for use with JAX/Haiku

from typing import Optional
import haiku as hk
import healpy as hp
import jax.numpy as jnp
import numpy as np


def build_neighbour_table(nside: int, nest: bool = True) -> jnp.ndarray:
    """Return neighbour lookup for every Healpix pixel.

    The table is shape (npix, 9). Each row contains the indices of the
    8 surrounding pixels plus the centre pixel in the middle slot. Pixels
    for which there are only 7 neighbours (identified with -1 by healpy)
    are filled with ``npix`` as consistent with the OG class, which is intended
    to index a zero padding row that is appended during the convolution.
    """
    npix = hp.nside2npix(nside)
    neighbours = np.empty((npix, 9), dtype=np.int32)
    for pix in range(npix):
        nb = hp.get_all_neighbours(nside, pix, nest=nest)
        nb = np.insert(nb, 4, pix) # include the centre pixel
        mask = nb == -1
        nb[mask] = npix # assign 'npix' like in spherical.py
        neighbours[pix] = nb

    return jnp.asarray(neighbours)

class HealpixConv(hk.Module):
    """Healpix 1-D convolution as described by Krachmalnicoff & Tomasi (2019).

    Parameters
    ----------
    neighbours:
        Precomputed neighbour lookup of shape (npix, kernel_size). Use
        :func:`build_neighbour_table` to generate it.
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    bias:
        Whether to learn an additive bias per output channel.
    name:
        Optional module name.
    """

    def __init__(
        self,
        neighbours: jnp.ndarray,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        if neighbours.ndim != 2:
            raise ValueError(
                "Neighbours array must have shape (npix, kernel_size)."
            )

        neighbours = jnp.asarray(neighbours)
        self.neighbours = neighbours
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = neighbours.shape[1]
        self.bias = bias
        self._pad_index = jnp.max(neighbours)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError("Expected input with shape (batch, npix, channels).")

        batch, npix, channels = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Input channels {channels} do not match expected {self.in_channels}."
            )

        # append a zero-valued pad pixel so sentinel indices gather zeros
        pad = jnp.zeros((batch, 1, channels), dtype=x.dtype)
        x_padded = jnp.concatenate([x, pad], axis=1)

        gathered = jnp.take(x_padded, self.neighbours, axis=1)
        # gathered shape: (batch, npix, kernel_size, channels)

        w = hk.get_parameter(
            "w",
            shape=(self.out_channels, self.kernel_size, self.in_channels),
            init=hk.initializers.VarianceScaling(),
        )

        # need to sit down and assess this einsum
        y = jnp.einsum("okc,bnkc->bno", w, gathered)

        if self.bias:
            b = hk.get_parameter(
                "b", shape=(self.out_channels,), init=hk.initializers.Constant(0.0)
            )
            y = y + b

        return y


class HealpixDown(hk.Module):
    """Average pooling layer on the Healpix sphere."""

    def __init__(self, groups: jnp.ndarray, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        if groups.ndim != 2:
            raise ValueError("Groups array must have shape (npix_coarse, n_children).")
        self.groups = groups

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError("Expected input with shape (batch, npix, channels).")

        gathered = jnp.take(x, self.groups, axis=1)
        # gathered: (batch, npix_coarse, n_children, channels)
        return jnp.mean(gathered, axis=2)
