"""Healpix-based embedding networks implemented in Haiku/JAX."""

from __future__ import annotations

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

from dipolesbi.tools.healsphere_conv import HealpixConv, HealpixDown, build_neighbour_table
import healpy as hp
import numpy as np


class HpCNNEmbedding(hk.Module):
    """CNN embedding for Healpix maps, ported from sphericalCNN."""

    def __init__(
        self,
        nside: int,
        in_channels: int = 1,
        out_channels_per_layer: Optional[list[int]] = None,
        nest: bool = True,
        n_blocks: Optional[int] = None,
        n_mlp_neurons: int = 64,
        n_mlp_layers: int = 2,
        output_dim: int = 32,
        dropout_rate: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        if n_blocks is None:
            n_blocks = int(np.log2(nside))
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")

        if out_channels_per_layer is None:
            out_channels_per_layer = [8, 16, 32, 64, 128, 256]
        out_channels_per_layer = out_channels_per_layer[:n_blocks]
        if len(out_channels_per_layer) < n_blocks:
            excess = n_blocks - len(out_channels_per_layer)
            out_channels_per_layer += [out_channels_per_layer[-1]] * excess

        self.nside = nside
        self.in_channels = in_channels
        self.out_channels_per_layer = out_channels_per_layer
        self.n_blocks = n_blocks
        self.nest = nest
        self.n_mlp_neurons = n_mlp_neurons
        self.n_mlp_layers = n_mlp_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Precompute neighbour tables and pooling groups per level
        self._neighbour_tables = []
        self._pool_tables = []
        cur_nside = nside
        for _ in range(n_blocks):
            neighbours = build_neighbour_table(cur_nside, nest)
            self._neighbour_tables.append(neighbours)

            parent_indices = _build_pool_groups(cur_nside, nest)
            self._pool_tables.append(parent_indices)
            cur_nside //= 2

    def __call__(self, x: jnp.ndarray, *, is_training: bool = True) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, npix, channels),"
                f" got {x.shape}"
            )

        z = x
        in_ch = self.in_channels
        for neighbours, pool_indices, out_ch in zip(
            self._neighbour_tables, self._pool_tables, self.out_channels_per_layer
        ):
            conv = HealpixConv(neighbours, in_ch, out_ch)
            z = conv(z)
            z = jax.nn.relu(z)
            pool = HealpixDown(pool_indices)
            z = pool(z)
            in_ch = out_ch

        z = jnp.reshape(z, (z.shape[0], -1))
        if self.dropout_rate > 0.0 and is_training:
            z = hk.dropout(hk.next_rng_key(), self.dropout_rate, z)

        mlp = hk.nets.MLP(
            [self.n_mlp_neurons] * (self.n_mlp_layers - 1) + [self.output_dim],
            activation=jax.nn.relu,
            name="hp_mlp",
        )
        return mlp(z)


def _build_pool_groups(nside: int, nest: bool) -> jnp.ndarray:
    if nside <= 1:
        raise ValueError("Cannot down-sample NSIDE <= 1")

    coarse_nside = nside // 2
    coarse_npix = hp.nside2npix(coarse_nside)
    coarse_pixels = np.arange(coarse_npix)
    nested = hp.ring2nest(coarse_nside, coarse_pixels) if not nest else coarse_pixels

    groups = []
    for pix in nested:
        child_nest = 4 * pix + np.arange(4)
        if nest:
            child_ring = child_nest
        else:
            child_ring = hp.nest2ring(nside, child_nest)
        groups.append(child_ring)
    return jnp.asarray(groups)
