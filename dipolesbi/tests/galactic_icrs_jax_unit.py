import numpy as np
import jax.numpy as jnp
from catsim.utils.healsphere import SkyCoord
import astropy.units as u
from dipolesbi.tools.coordinates import _galactic_to_equatorial_vec_jax
import healpy as hp


def test_galactic_to_equatorial_vector_matches_astropy():
    # Deterministic sample points covering a range of angles.
    gal_lon = np.array([0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
    gal_lat = np.array([-60.0, -30.0, 0.0, 15.0, 45.0, 80.0])

    gal = SkyCoord(gal_lon, gal_lat, unit=u.deg, frame="galactic")
    eq = gal.transform_to("icrs")
    eq_lon = eq.ra.deg  # pyright: ignore[reportOptionalMemberAccess]
    eq_lat = eq.dec.deg  # pyright: ignore[reportOptionalMemberAccess]
    # vec_astropy = _ang2vec_np(eq_lon, eq_lat)

    vec_jax = _galactic_to_equatorial_vec_jax(
        jnp.asarray(gal_lon),
        jnp.asarray(gal_lat)
    )

    # go back to lonlat
    eq_lon_jax, eq_lat_jax = hp.vec2ang(vec_jax, lonlat=True)

    np.testing.assert_allclose(
        np.asarray(eq_lon_jax),
        np.asarray(eq_lon),
        atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(eq_lat_jax),
        np.asarray(eq_lat),
        atol=1e-4
    )
