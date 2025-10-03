import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import healpy as hp


def downgrade_ignore_nan(
    map_in: NDArray,
    mask_in: NDArray,
    nside_out: int
) -> tuple[NDArray, NDArray]:
    """Downgrade NEST-ordered maps/masks, ignoring masked pixels.

    Args:
        map_in: Array of shape ``(npix,)`` or ``(batch, npix)``.
        mask_in: Boolean array matching ``map_in``; ``False`` marks masked pixels.
        nside_out: Target HEALPix nside (must be a power-of-two divisor).
    Returns:
        Tuple ``(map_out, mask_out)`` matching the input batch shape.
    """

    map_arr = np.asarray(map_in)
    mask_arr = np.asarray(mask_in)

    squeezed_input = map_arr.ndim == 1

    if map_arr.ndim == 2 and mask_arr.ndim == 1:
        assert mask_arr.size == map_arr.shape[1]
        mask_arr = np.broadcast_to(mask_arr, map_arr.shape)

    if squeezed_input and mask_arr.ndim == 2:
        assert mask_arr.shape[0] == 1 and mask_arr.shape[1] == map_arr.size
        mask_arr = mask_arr[0]

    assert map_arr.shape == mask_arr.shape

    if squeezed_input:
        map_arr = map_arr[None, :]
        mask_arr = mask_arr[None, :]

    assert map_arr.ndim == 2, 'map_in must be 1D or 2D (batch, npix).'

    orig_dtype = map_arr.dtype
    work_map = map_arr.astype(np.float64, copy=False)
    work_mask = mask_arr.astype(bool, copy=False)

    work_map = np.where(work_mask, work_map, np.nan)

    npix = work_map.shape[-1]
    cur_nside = hp.npix2nside(npix)
    assert cur_nside >= nside_out, 'Output nside must be lower than input nside.'
    ratio = cur_nside // nside_out
    assert (cur_nside % nside_out) == 0 and (ratio & (ratio - 1)) == 0

    out_map = work_map
    out_mask = work_mask

    while cur_nside > nside_out:
        cur_npix = hp.nside2npix(cur_nside)
        cur_npix //= 4

        out_map = out_map.reshape(out_map.shape[0], cur_npix, 4)
        out_mask = out_mask.reshape(out_mask.shape[0], cur_npix, 4)

        valid_counts = out_mask.sum(axis=-1)
        summed = np.where(out_mask, out_map, 0.0).sum(axis=-1)

        out_mask = valid_counts > 0
        out_map = summed
        out_map[~out_mask] = np.nan

        cur_nside //= 2

    if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
        out_map = out_map.astype(orig_dtype, copy=False)

    if squeezed_input:
        out_map = out_map[0]
        out_mask = out_mask[0]

    return out_map, out_mask


def downgrade_ignore_nan_jax(
    map_in: jnp.ndarray,
    mask_in: jnp.ndarray,
    nside_out: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX equivalent of ``downgrade_ignore_nan`` supporting 1D/2D inputs."""

    map_arr = jnp.asarray(map_in)
    mask_arr = jnp.asarray(mask_in, dtype=bool)

    squeezed_input = map_arr.ndim == 1

    if map_arr.ndim == 2 and mask_arr.ndim == 1:
        mask_arr = jnp.broadcast_to(mask_arr, map_arr.shape)

    if squeezed_input and mask_arr.ndim == 2:
        mask_arr = mask_arr[0]

    if squeezed_input:
        map_arr = map_arr[None, :]
        mask_arr = mask_arr[None, :]

    assert map_arr.shape == mask_arr.shape, 'map and mask shapes must agree.'
    assert map_arr.ndim == 2, 'map_in must be 1D or 2D (batch, npix).'

    orig_dtype = map_arr.dtype
    if jnp.issubdtype(orig_dtype, jnp.floating):
        work_dtype = orig_dtype
    else:
        work_dtype = jnp.float32

    work_map = map_arr.astype(work_dtype)
    work_mask = mask_arr.astype(bool)

    work_map = jnp.where(work_mask, work_map, jnp.nan)

    npix = work_map.shape[-1]
    cur_nside = hp.npix2nside(int(npix))
    assert cur_nside >= nside_out, 'Output nside must be lower than input nside.'
    ratio = cur_nside // nside_out
    assert (cur_nside % nside_out) == 0 and (ratio & (ratio - 1)) == 0

    out_map = work_map
    out_mask = work_mask

    while cur_nside > nside_out:
        cur_npix = hp.nside2npix(cur_nside)
        cur_npix //= 4

        out_map = out_map.reshape(out_map.shape[0], cur_npix, 4)
        out_mask = out_mask.reshape(out_mask.shape[0], cur_npix, 4)

        valid_counts = jnp.sum(out_mask, axis=-1)
        summed = jnp.sum(jnp.where(out_mask, out_map, 0.0), axis=-1)

        out_mask = valid_counts > 0
        out_map = jnp.where(out_mask, summed, jnp.nan)

        cur_nside //= 2

    if jnp.issubdtype(orig_dtype, jnp.floating) and work_dtype != orig_dtype:
        out_map = out_map.astype(orig_dtype)

    if squeezed_input:
        out_map = out_map[0]
        out_mask = out_mask[0]

    return out_map, out_mask

def _split_len(L: int, n: int) -> list[int]:
    """Split length L into n near-equal positive chunks that sum to L."""
    q, r = divmod(L, n)
    return [q + (1 if i < r else 0) for i in range(n)]

def split_off_details(
    initial_nside: int,
    output_nside: int
) -> list[tuple[int, int]]:
    ns = initial_nside
    nkeep_ndrop = []
    while ns > output_nside:
        n_pix = 12 * ns**2
        n_details = 0.75 * n_pix 
        n_keep = int(n_pix - n_details)
        n_drop = int(n_details)
        nkeep_ndrop.append((n_keep, n_drop))
        ns //= 2
    return nkeep_ndrop

def build_funnel_steps(
    n_coarse: int,
    detail_lengths: list[int],
    n_chunks: int | list[int] = 1,   # e.g. 2 => halves; can pass per-level list
) -> list[tuple[np.ndarray, int, int]]:
    """
    State starts as [coarse | d1 | d2 | ...].
    If n_chunks > 1, each detail dℓ is split into n_chunks pieces and we peel them
    in order: d1[0], d1[1], ..., then d2[0], d2[1], ... (i.e., fully peel d1 in pieces,
    then d2 in pieces, etc.).
    Returns list of (perm, n_keep, n_drop) per funnel step.
    """
    if isinstance(n_chunks, int):
        n_chunks = [n_chunks] * len(detail_lengths)
    assert len(n_chunks) == len(detail_lengths)

    # Build initial remaining lengths in *current* order:
    # [coarse, d1_0, d1_1, ..., d2_0, d2_1, ..., ...]
    remaining = [n_coarse]
    for L, k in zip(detail_lengths, n_chunks):
        chunks = _split_len(L, k) if k > 1 else [L]
        remaining.extend(chunks)

    steps: list[tuple[np.ndarray, int, int]] = []
    while len(remaining) > 1:
        starts = np.cumsum([0] + remaining[:-1])
        blocks = [np.arange(starts[i], starts[i] + remaining[i]) for i in range(len(remaining))]

        # Drop the first detail *chunk* (immediately after coarse).
        drop_block = blocks[1]
        keep_blocks = [blocks[0]] + blocks[2:]   # coarse + all remaining detail chunks

        keep_idx = np.concatenate(keep_blocks, 0)
        drop_idx = drop_block
        dim_before = sum(remaining)

        # Build permutation [keep | drop] and sanity-check it.
        perm = np.concatenate([keep_idx, drop_idx], 0)
        assert perm.shape[0] == dim_before
        assert np.array_equal(np.sort(perm), np.arange(dim_before))

        steps.append((perm, int(keep_idx.size), int(drop_idx.size)))

        # After dropping this chunk, the new state is [coarse | (rest of chunks)]
        remaining = [remaining[0]] + remaining[2:]

    return steps

# def build_funnel_steps(
#         n_coarse: int, 
#         detail_lengths: list[int]
# ) -> list[tuple[jnp.ndarray, int, int]]:
#     """Return a list of (perm, n_keep, n_drop) for a state initially
#        [coarse | d1 | d2 | ...], dropping d1 then d2 ..."""
#     remaining = [n_coarse] + list(detail_lengths)
#     steps = []
#     while len(remaining) > 1:
#         starts = np.cumsum([0] + remaining[:-1])
#         blocks = [np.arange(starts[i], starts[i] + remaining[i]) for i in range(len(remaining))]
#         keep_blocks = [blocks[0]] + blocks[2:]   # coarse + d2..end
#         drop_block  = blocks[1]                  # current d1
#         keep_idx = np.concatenate(keep_blocks, 0)
#         drop_idx = drop_block
#         # sanity: permutation must be 0..(dim-1)
#         dim_before = sum(remaining)
#         perm = np.concatenate([keep_idx, drop_idx], 0)
#         assert perm.shape[0] == dim_before
#         assert np.array_equal(np.sort(perm), np.arange(dim_before))
#         steps.append((jnp.asarray(perm), int(keep_idx.size), int(drop_idx.size)))
#         # after dropping d1, the state is [coarse | d2 | ...]
#         remaining = [remaining[0]] + remaining[2:]
#     return steps

def permute_within_types(cur_dim: int, n_coarse: int, seed: int):
    c = min(n_coarse, cur_dim)
    d = cur_dim - c
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)
    p1 = jax.random.permutation(k1, c)
    p2 = jnp.array([], dtype=jnp.int32) if d == 0 else (jax.random.permutation(k2, d) + c)
    return jnp.concatenate([p1, p2], axis=0)

def make_latent_dims(dim0: int, n_layers: int, reduction_factor: float):
    dims = []
    d = dim0
    for _ in range(n_layers):
        ld = int(reduction_factor * d)
        dims.append(ld)
        d = ld
    return dims  # list[int]

def permute_within_strata(length: int, n_strata: int, seed: int):
    """
    Returns a permutation of [0..length-1] that keeps modulo-`n_strata` groups
    contiguous but randomly permutes elements *within* each group.

    No boolean indexing; just a single argsort on integer keys:
      key = gid * BIG + rnd
    where gid = i % n_strata and rnd is a per-index random int < BIG.
    """
    idx = jnp.arange(length)
    gid = jnp.mod(idx, n_strata).astype(jnp.int32)      # group id 0..n_strata-1

    key = jax.random.PRNGKey(seed)
    rnd = jax.random.randint(key, (length,), 0, 2**31 - 1, dtype=jnp.int32)

    BIG = jnp.int32(2**31 - 1)                           # ensure rnd < BIG
    sort_key = gid * BIG + rnd                           # group-dominant key

    order = jnp.argsort(sort_key)                        # groups in gid order; shuffled within
    return order

def build_layer_perms(latent_dims: list[int], n_strata: int = 12, base_seed: int = 0):
    """
    One permutation per layer, acting on the kept length only.
    Each perm length = latent_dims[i]. JAX-safe.
    """
    return [permute_within_strata(ld, n_strata, base_seed + i)
            for i, ld in enumerate(latent_dims)]

def get_healpix_superpixels(nside: int, super_nside: int = 1) -> list[jnp.ndarray]:
    block_size = nside**2
    super_npix = 12 * super_nside**2
    blocks = [jnp.arange(b*block_size, (b+1)*block_size) for b in range(super_npix)]
    return blocks

def interleave_blocks(blocks):
    # blocks: list of 12 arrays, each length block_size
    # Returns a single array of length N that alternates 1-by-1 across blocks.
    rows = jnp.stack(blocks, axis=0) # [12, block_size]
    interleaved = rows.T.reshape(-1) # [block_size*12], column-major
    return interleaved

def first_layer_stratifying_perm(latent_dim: int, blocks: list[jnp.ndarray]):
    """
    blocks: list of B disjoint index arrays, each of equal length block_size.
            For HEALPix NEST at fixed nside, B=12 and block_size = nside**2.
    Returns a permutation 'perm' s.t. the first 'latent_dim' indices are
    round-robin across blocks, and the remainder keeps the same interleaved order.
    """
    # Interleave 1-by-1 across blocks: shape [B, block_size] -> [dim]
    rows = jnp.stack(blocks, axis=0)      # [B, block_size]
    interleaved = rows.T.reshape(-1)      # [dim], column-major interleave

    # Just split the interleaved sequence
    take = interleaved[:latent_dim]       # stratified head
    rest = interleaved[latent_dim:]       # the remainder
    return jnp.concatenate([take, rest], axis=0)
