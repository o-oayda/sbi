from typing import Any, Callable
from catsim import RacsLow3Config
import numpy as np
from numpy.typing import NDArray
from dipolesbi.tools.model_config_io import load_model_config
from dipolesbi.tools.neural_flows import NeuralFlow
import jax.numpy as jnp
import jax
from dipolesbi.tools.priors_jax import DipolePriorJax


def get_lnlike_data_prior(
    output_dir: str,
    round_number: int
) -> dict[str, Any]:
    '''
    Return summary dict containing:
    ```
    summary_dict = {
        'lnlike': lnlike_jax,
        'data': {
            'x0': x0, # downscaled if chosen
            'mask': mask
        },
        'prior': prior_jax
    }
    ```
    '''
    checkpoint_path = f'{output_dir}/nflow_checkpoint_r{round_number}.npz'
    x0, mask = get_x0_mask_from_json(output_dir)
    lnlike_jax = get_lnlike_from_chkpt(checkpoint_path, x0, mask)
    prior_jax = get_prior_from_chkpt(checkpoint_path)
    summary_dict = {
        'lnlike': lnlike_jax,
        'data': {
            'x0': x0, # downscaled if chosen
            'mask': mask
        },
        'prior': prior_jax
    }
    return summary_dict

def get_lnlike_from_chkpt(
        checkpoint_path: str,
        x0: NDArray,
        x0_mask: NDArray[np.bool_]
) -> Callable[[dict[str, jnp.ndarray]], jnp.ndarray]:
    '''
    :checkpoint_path: str path to the location of the nflow_checkpoint after
        rounds of SSNLE inference.     
    :param x0: The D in P(D | Theta, M), i.e. the data we are learning on.
    :param x0_mask: The boolean mask on the data vector defining which
        elements are to be passed in the likelihood function.

    z0 (the transformed x0) is needed for closure in lnlike_jax, so we must
    pass in x0 here and not in the output likelihood function.
    '''
    neural_flow, transform_cfg = NeuralFlow.from_checkpoint(checkpoint_path)
    data_transform = transform_cfg.data_transform_config.data_transform
    theta_transform = transform_cfg.theta_transform_config.theta_transform

    if data_transform is not None:
        (z0, z0_mask), log_det_jac = data_transform(x0, x0_mask)
    else:
        z0, z0_mask = x0, x0_mask
        log_det_jac = np.zeros((1,), dtype=np.float32)
    
    z0 = jax.device_put(z0)
    zmask0 = jax.device_put(z0_mask)
    log_det_jac = jax.device_put(log_det_jac)

    def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        assert theta_transform is not None
        theta, _ = theta_transform(params, in_ns=True)

        log_like = neural_flow.evaluate_lnlike(
            theta[None, :], 
            z0,
            mask=zmask0
        )

        log_like += log_det_jac
        return log_like.squeeze()

    return lnlike_jax

def get_prior_from_chkpt(
        checkpoint_path: str,
) -> DipolePriorJax:
    _, transform_cfg = NeuralFlow.from_checkpoint(checkpoint_path)
    prior = transform_cfg.theta_transform_config.prior
    if prior is None:
        raise Exception('No prior found in checkpoint.')
    return prior

def get_x0_mask_from_json(
        out_dir: str
) -> tuple[NDArray, NDArray]:
    '''
    :return: x0, mask tuple.
    '''
    reference = np.load(f'{out_dir}/reference_observation.npz')
    return reference['x0'], reference['mask']
