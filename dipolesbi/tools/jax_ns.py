from typing import Callable, Optional
from anesthetic import NestedSamples
from blackjax.types import PRNGKey
import jax
from jax import numpy as jnp
import blackjax
from tqdm import tqdm
from dipolesbi.tools.priors_jax import JaxPrior
from dipolesbi.tools.ui import MultiRoundInfererUI

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
