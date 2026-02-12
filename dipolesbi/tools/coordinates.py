import jax.numpy as jnp


def _ang2vec_jax(lon_deg: jnp.ndarray, lat_deg: jnp.ndarray) -> jnp.ndarray:
    lon = jnp.deg2rad(lon_deg)
    lat = jnp.deg2rad(lat_deg)
    cos_lat = jnp.cos(lat)
    return jnp.stack(
        [cos_lat * jnp.cos(lon), cos_lat * jnp.sin(lon), jnp.sin(lat)],
        axis=-1
    )

def _galactic_to_equatorial_vec_jax(l_deg: jnp.ndarray, b_deg: jnp.ndarray) -> jnp.ndarray:
    # IAU 2000/2009 Galactic->ICRS rotation matrix (astropy default).
    # Source: ICRS2G routine (lists both directions) in astro-fortran.
    # https://jacobwilliams.github.io/astro-fortran/proc/icrs2g.html
    r_gal_to_icrs = jnp.asarray(
        [
            [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
            [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
            [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669],
        ],
        dtype=jnp.float64
    )
    v_gal = _ang2vec_jax(l_deg, b_deg)
    return v_gal @ r_gal_to_icrs
