import operator
from typing import Callable, Optional
from anesthetic import NestedSamples
from blackjax.types import PRNGKey
import jax
from jax import numpy as jnp
import blackjax
from numpy.typing import NDArray
from tqdm import tqdm
from dipolesbi.tools.neural_flows import NeuralFlow
from dipolesbi.tools.priors_jax import DipolePriorJax, JaxPrior
from dipolesbi.tools.ui import MultiRoundInfererUI
import numpy as np
import os
import matplotlib.pyplot as plt


class JaxNestedSampler:
    def __init__(
            self,
            lnlike: Callable, 
            prior: JaxPrior, 
            ui: Optional[MultiRoundInfererUI] = None
    ) -> None:
        self.lnlike = lnlike
        self.prior = prior
        self.ndim = self.prior.ndim
        self.ui = ui
        if self.ui:
            self.print_func = self.ui.log
        else:
            self.print_func = print

    def setup(
            self,
            rng_key: PRNGKey, 
            n_live: int = 500,
            n_delete: int = 50
    ) -> None:
        n_inner_steps = self.ndim * 5
        self.n_delete = n_delete
        self.n_live = n_live
        self.rng_key, self.prior_key = jax.random.split(rng_key)
        self.particles = self.prior.get_initial_live_samples(self.prior_key, n_live)

        self.nested_sampler = blackjax.nss(
            logprior_fn=self.prior.log_prob,
            loglikelihood_fn=self.lnlike,
            num_delete=n_delete,
            num_inner_steps=n_inner_steps,
        )

        self._jit_functions()

    def _jit_functions(self) -> None:
        self.init_fn = jax.jit(self.nested_sampler.init)
        self.step_fn = jax.jit(self.nested_sampler.step)

    def run(self) -> NestedSamples:
        live = self.init_fn(self.particles)
        dead = []

        if self.ui:
            self.ui.begin_progress(total=None, description='Dead points')
            while not live.logZ_live - live.logZ < -3:  # Convergence criterion
                self.rng_key, subkey = jax.random.split(self.rng_key, 2)
                live, dead_info = self.step_fn(subkey, live)
                dead.append(dead_info)
                self.ui.update_progress(self.n_delete)
            self.ui.end_progress()
        else:
            with tqdm(desc="Dead points", unit=" dead points") as pbar:
                while not live.logZ_live - live.logZ < -3:  # Convergence criterion
                    self.rng_key, subkey = jax.random.split(self.rng_key, 2)
                    live, dead_info = self.step_fn(subkey, live)
                    dead.append(dead_info)
                    pbar.update(self.n_delete)

        dead = blackjax.ns.utils.finalise(live, dead)
        columns = self.prior.simulator_kwargs
        data = jnp.vstack([dead.particles[key] for key in columns]).T # type: ignore

        self.nested_samples = NestedSamples(
            data,
            logL=dead.loglikelihood,
            logL_birth=dead.loglikelihood_birth,
            columns=columns,
            logzero=jnp.nan,
        )

        self.evidence_str = (
            f"[cyan]ln Z: {self.nested_samples.logZ():.2f} "
            f"± {self.nested_samples.logZ(100).std():.2f}[/cyan]" # type: ignore
        )
        self.print_func(self.evidence_str)
        
        return self.nested_samples

def merge_priors(
        prior_A: DipolePriorJax, 
        prior_B: DipolePriorJax
) -> DipolePriorJax:
    # if dataset B is given, create super-prior that contains all parameters
    # from prior A and prior B
    prior_A_dict = prior_A.prior_dict; prior_B_dict = prior_B.prior_dict
    super_prior = prior_A_dict
    for key, val in prior_B_dict.items():
        if key not in super_prior.keys():
            super_prior[key] = val
        else:
            # assert the priors are identical at common keys
            for chk in ['dist_type', 'low_range', 'high_range']:
                assert super_prior[key][chk] == prior_B_dict[key][chk], (
                    "Shared paramaters have inconsistent priors! "
                    f"Mismatch at {key}."
                )

    super_prior_instance = DipolePriorJax.from_prior_dict(super_prior)
    return super_prior_instance

def run_ns_from_chkpt(
    path_to_chkpt: str,
    data: NDArray,
    mask: NDArray[np.bool_],
    jax_key: PRNGKey,
    lnlike_B: Optional[
        Callable[[dict[str, jnp.ndarray]], jnp.ndarray]
    ] = None,
    prior_B: Optional[DipolePriorJax] = None,
    data_B: Optional[NDArray] = None,
    save_dir: Optional[str] = None
) -> NestedSamples:
    '''
    :param lnlike_B: If passed, compute a joint evidence
        Z_AB = int L_A L_B prior d theta
    '''
    chkpt_src = os.path.join(*path_to_chkpt.split('/')[:-1])
    neural_flow, transform_cfg = NeuralFlow.from_checkpoint(path_to_chkpt)
    data_transform = transform_cfg.data_transform_config.data_transform
    theta_transform = transform_cfg.theta_transform_config.theta_transform
    prior = transform_cfg.theta_transform_config.prior
    assert prior is not None

    # add batch dimension if none provided
    if data.ndim == 1:
        data = data[None, :]

    if data_transform is not None:
        (z0, z0_mask), log_det_jac = data_transform(data, mask)
    else:
        z0, z0_mask = data, mask
        log_det_jac = np.zeros((1,), dtype=np.float32)

    z0 = jax.device_put(z0)
    zmask0 = jax.device_put(z0_mask)
    log_det_jac = jax.device_put(log_det_jac)

    if lnlike_B is not None:
        assert prior_B is not None; assert data_B is not None
        prior_A = prior
        
        # since the jax ns expects simulator_kwargs as the keys for the params
        # dict, we use those as the keys
        super_prior = merge_priors(prior_A, prior_B)
        prior_A_keys = list(prior_A.simulator_kwargs)
        prior_A_getter = operator.itemgetter(*prior_A_keys)
        prior_B_keys = list(prior_B.simulator_kwargs)
        prior_B_getter = operator.itemgetter(*prior_B_keys)

        def joint_lnlike_jax(super_params: dict[str, jnp.ndarray]) -> jnp.ndarray:
            assert theta_transform is not None
            prior_A_samples = prior_A_getter(super_params)
            theta_A, _ = theta_transform(prior_A_samples, in_ns=True)

            log_like_A = neural_flow.evaluate_lnlike(
                theta_A[None, :], 
                z0,
                mask=zmask0
            )
            log_like_A += log_det_jac
            
            # now evaluate on B
            prior_B_samples = prior_B_getter(super_params)
            prior_B_samplesdict = dict(zip(prior_B_keys, prior_B_samples))
            log_like_B = lnlike_B(prior_B_samplesdict)

            log_like = log_like_A + log_like_B

            return log_like.squeeze()

        jax_ns = JaxNestedSampler(
            joint_lnlike_jax,
            super_prior,
            ui=None
        )

    else:
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

        jax_ns = JaxNestedSampler(
            lnlike_jax, 
            prior,
            ui=None
        )

    jax_ns.setup(jax_key, n_live=1000, n_delete=200)
    nested_samples = jax_ns.run()

    if save_dir is None:
        save_dir = chkpt_src

    kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
    plt.figure()
    nested_samples.plot_2d(
        super_prior.simulator_kwargs if lnlike_B is not None else prior.simulator_kwargs,
        kinds=kinds
    )
    plt.savefig(
        f'{save_dir}/final_samples.pdf',
        bbox_inches='tight'
    )
    plt.close()
    nested_samples.to_csv(f'{save_dir}/nested_samples.csv')

    return nested_samples
