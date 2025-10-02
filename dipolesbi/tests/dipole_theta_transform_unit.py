import jax.numpy as jnp
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.transforms import DipoleThetaTransform
from dipolesbi.tools.utils import PytreeAdapter


def _build_transform():
    prior = DipolePriorJax()
    prior.change_kwarg('N', 'mean_density')

    transform = DipoleThetaTransform(prior=prior, method='cartesian')

    theta_stats = {
        'mean_density': jnp.array([1.0, 1.5, 2.0]),
        'observer_speed': jnp.array([200.0, 210.0, 220.0]),
        'dipole_longitude': jnp.array([0.0, 45.0, 90.0]),
        'dipole_latitude': jnp.array([0.0, 15.0, -30.0]),
    }
    transform.compute_mean_and_std(theta_stats)
    return transform


def _build_zscore_transform():
    prior = DipolePriorJax()
    prior.change_kwarg('N', 'mean_density')

    transform = DipoleThetaTransform(prior=prior, method='zscore')

    theta_stats = {
        'mean_density': jnp.array([1.0, 1.5, 2.0]),
        'observer_speed': jnp.array([200.0, 210.0, 220.0]),
        'dipole_longitude': jnp.array([0.0, 45.0, 90.0]),
        'dipole_latitude': jnp.array([0.0, 15.0, -30.0]),
    }
    transform.compute_mean_and_std(theta_stats)
    return transform


def test_forward_cartesian_batch_shapes_and_inverse_consistency():
    transform = _build_transform()

    theta_batch = {
        'mean_density': jnp.array([1.2, 1.8]),
        'observer_speed': jnp.array([205.0, 215.0]),
        'dipole_longitude': jnp.array([30.0, 60.0]),
        'dipole_latitude': jnp.array([10.0, -20.0]),
    }

    z, logdet = transform.forward_and_log_det(theta_batch)

    assert z.shape == (2, 5)
    assert logdet.shape == (2,)

    theta_recon, inv_logdet = transform.inverse_and_log_det(z)

    original_array = transform.adapter.to_array(theta_batch)
    assert theta_recon.shape == original_array.shape
    assert jnp.allclose(theta_recon, original_array, atol=1e-6)
    assert jnp.allclose(logdet + inv_logdet, jnp.zeros_like(logdet), atol=1e-6)


def test_forward_cartesian_single_sample_in_ns():
    transform = _build_transform()

    theta_single_tree = {
        'mean_density': jnp.array(1.4),
        'observer_speed': jnp.array(212.0),
        'dipole_longitude': jnp.array(75.0),
        'dipole_latitude': jnp.array(-5.0),
    }

    z_single, logdet_single = transform.forward_and_log_det(theta_single_tree, in_ns=True)

    assert z_single.shape == (5,)
    assert jnp.isscalar(logdet_single) or logdet_single.shape == ()

    theta_recon_single, inv_logdet_single = transform.inverse_and_log_det(z_single)

    single_array = transform.adapter.ravel(theta_single_tree)
    assert theta_recon_single.shape == single_array.shape
    assert jnp.allclose(theta_recon_single, single_array, atol=1e-6)
    assert jnp.allclose(logdet_single + inv_logdet_single, 0.0, atol=1e-6)


def test_forward_zscore_batch_round_trip():
    transform = _build_zscore_transform()

    theta_batch = {
        'mean_density': jnp.array([1.1, 1.9]),
        'observer_speed': jnp.array([202.0, 218.0]),
        'dipole_longitude': jnp.array([25.0, 80.0]),
        'dipole_latitude': jnp.array([5.0, -12.0]),
    }

    z, logdet = transform.forward_and_log_det(theta_batch)

    assert z.shape == (2, 4)
    assert logdet.shape == (2,)

    theta_recon, inv_logdet = transform.inverse_and_log_det(z)

    original_array = transform.adapter.to_array(theta_batch)
    assert theta_recon.shape == original_array.shape
    assert jnp.allclose(theta_recon, original_array, atol=1e-6)
    assert jnp.allclose(logdet + inv_logdet, jnp.zeros_like(logdet), atol=1e-6)


def test_forward_zscore_single_in_ns():
    transform = _build_zscore_transform()

    theta_single = {
        'mean_density': jnp.array(1.6),
        'observer_speed': jnp.array(208.0),
        'dipole_longitude': jnp.array(55.0),
        'dipole_latitude': jnp.array(-7.5),
    }

    z_single, logdet_single = transform.forward_and_log_det(theta_single)

    assert z_single.shape == (4,)
    assert jnp.isscalar(logdet_single) or logdet_single.shape == ()

    theta_recon_single, inv_logdet_single = transform.inverse_and_log_det(z_single)

    single_array = transform.adapter.ravel(theta_single)
    assert theta_recon_single.shape == single_array.shape
    assert jnp.allclose(theta_recon_single, single_array, atol=1e-6)
    assert jnp.allclose(logdet_single + inv_logdet_single, 0.0, atol=1e-6)
