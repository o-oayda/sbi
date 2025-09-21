import jax
import jax.numpy as jnp

from dipolesbi.tools.utils import PytreeAdapter


def _example_tree():
    return {
        'mean_density': jnp.array(1.23),
        'observer_speed': jnp.array([200.0, 201.0]),
        'dipole_longitude': jnp.array([[10.0, 20.0], [30.0, 40.0]]),
        'dipole_latitude': jnp.array(-15.0),
    }


def test_ravel_unravel_roundtrip():
    tree = _example_tree()
    adapter = PytreeAdapter(tree)

    flat = adapter.ravel(tree)
    restored = adapter.unravel(flat)

    for key in tree:
        assert jnp.array_equal(restored[key], tree[key])


def test_ravel_unravel_batch_roundtrip():
    tree = _example_tree()
    adapter = PytreeAdapter(tree)

    batch_tree = jax.vmap(lambda offset: {
        'mean_density': tree['mean_density'] + offset,
        'observer_speed': tree['observer_speed'] + offset,
        'dipole_longitude': tree['dipole_longitude'] + offset,
        'dipole_latitude': tree['dipole_latitude'] + offset,
    })(jnp.array([0.0, 1.0, 2.0]))

    flat = adapter.to_array(batch_tree)
    restored = adapter.to_pytree(flat)

    for key in tree:
        assert jnp.array_equal(restored[key], batch_tree[key])


def test_flat_view_slices():
    tree = _example_tree()
    adapter = PytreeAdapter(tree)

    batch_tree = jax.vmap(lambda shift: {
        'mean_density': tree['mean_density'] + shift,
        'observer_speed': tree['observer_speed'] + shift,
        'dipole_longitude': tree['dipole_longitude'] + shift,
        'dipole_latitude': tree['dipole_latitude'] + shift,
    })(jnp.array([0.0, 1.0]))

    flat = adapter.to_array(batch_tree)

    mean_view = adapter.flat_view(flat, 'mean_density')
    speed_view = adapter.flat_view(flat, 'observer_speed')
    lon_view = adapter.flat_view(flat, 'dipole_longitude')
    lat_view = adapter.flat_view(flat, 'dipole_latitude')

    assert mean_view.shape == (2,)
    assert speed_view.shape == (2, 2)
    assert lon_view.shape == (2, 2, 2)
    assert lat_view.shape == (2,)

    assert jnp.array_equal(mean_view, batch_tree['mean_density'])
    assert jnp.array_equal(speed_view, batch_tree['observer_speed'])
    assert jnp.array_equal(lon_view, batch_tree['dipole_longitude'])
    assert jnp.array_equal(lat_view, batch_tree['dipole_latitude'])
