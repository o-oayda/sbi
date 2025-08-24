import jax
import jax.numpy as jnp


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
