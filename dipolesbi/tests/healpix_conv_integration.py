import numpy as np
import torch
import healpy as hp
import jax
import jax.numpy as jnp
import haiku as hk
from dipolesbi.lib.torch_hp_cnn import sphericalConv, sphericalDown
from dipolesbi.tools.healsphere_conv import HealpixConv, HealpixDown, build_neighbour_table
from dipolesbi.tools.embedding_nets import _build_pool_groups


def test_jax_hpconv_matches_torch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    nside = 2
    in_channels = 1
    out_channels = 2
    npix = hp.nside2npix(nside)

    # Torch reference modules
    torch_conv = sphericalConv(nside, in_channels, out_channels, nest=True)
    torch_pool = sphericalDown(nside)
    torch_relu = torch.nn.ReLU()

    # JAX modules
    neighbours = build_neighbour_table(nside, nest=True)
    pool_groups = _build_pool_groups(nside, nest=True)

    def forward_fn(x):
        conv = HealpixConv(neighbours, in_channels, out_channels, name="conv")
        pool = HealpixDown(pool_groups, name="down")
        z = conv(x)
        z = jax.nn.relu(z)
        z = pool(z)
        return z

    hk_mod = hk.without_apply_rng(hk.transform(forward_fn))
    params = hk_mod.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, npix, in_channels), dtype=jnp.float32)
    )

    # Copy weights/bias from Torch to Haiku (note transpose to match layout)
    torch_w = torch_conv.conv.weight.detach().cpu().numpy()  # (out, in, kernel)
    torch_b = torch_conv.conv.bias.detach().cpu().numpy()
    params["conv"]["w"] = jnp.asarray(np.transpose(torch_w, (0, 2, 1)))
    if torch_b is not None:
        params["conv"]["b"] = jnp.asarray(torch_b)

    # Prepare identical inputs
    x = rng.standard_normal((3, npix)).astype(np.float32)
    x_jnp = x[..., None]
    x_torch = torch.from_numpy(x[:, None, :])  # (batch, channels, npix)

    torch_out = (
        torch_pool(torch_relu(torch_conv(x_torch)))
        .detach()
        .cpu()
        .numpy()
    )
    torch_out = np.transpose(torch_out, (0, 2, 1))  # (batch, npix_coarse, out_channels)

    jax_out = np.array(hk_mod.apply(params, x_jnp))

    # print(f'Jax: {jax_out}')
    # print(f'Torch: {torch_out}')

    # the gpu-side calculation by jax means fp errors accumulate,
    # so we need to be more generous with the comparison
    assert np.allclose(jax_out, torch_out, atol=1e-3, rtol=1e-3)
