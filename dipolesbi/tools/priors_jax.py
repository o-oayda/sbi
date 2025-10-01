from abc import ABC, abstractmethod
from typing import Callable
import jax
from blackjax.types import PRNGKey
from dipolesbi.tools.utils import PytreeAdapter, polar_logpdf_jax, sample_polar_jax
from jax import numpy as jnp


class JaxPrior(ABC):
    def __init__(self) -> None:
        self.device = None

    @property
    @abstractmethod
    def prior_dict(self) -> dict[str, dict]:
        '''
        Contains information about each variable and their sampling distribution.
        '''
        pass

    @property
    @abstractmethod
    def prior_names(self) -> list[str]:
        '''
        Controls the order of sampling, i.e. order of access to prior_dict.
        '''
        pass

    @property
    def ndim(self) -> int:
        return len(self.prior_dict)

    @property
    def low_ranges(self) -> jnp.ndarray:
        vals = [self.prior_dict[n]['low_range'] for n in self.prior_names]
        return jnp.asarray(vals, dtype=jnp.float32)

    @property
    def high_ranges(self) -> jnp.ndarray:
        vals = [self.prior_dict[n]['high_range'] for n in self.prior_names]
        return jnp.asarray(vals, dtype=jnp.float32)

    @property
    def simulator_kwargs(self) -> list[str]:
        return [self.prior_dict[name]['simulator_kwarg'] for name in self.prior_names]

    def change_kwarg(self, param_short_name: str, new_kwarg: str) -> None:
        self.prior_dict[param_short_name]['simulator_kwarg'] = new_kwarg

    def _construct_prior_dict(
            self,
            short_names: list[str],            
            simulator_kwargs: list[str],
            ranges: list[list[float]],
            sample_funcs: list[Callable],
            logpdf_funcs: list[Callable],
            dist_types: list[str]
            
    ) -> dict[str, dict]:
        '''
        Format like {
            '<short_name>': {
                '<simulator_kwarg>': str,
                '<low_range>': float,
                '<high_range': float,
                'dist': Distribution    
            },
            ...
        }
        '''
        prior_dict = {}
        for i, name in enumerate(short_names):
            prior_dict[name] = {
                'simulator_kwarg': simulator_kwargs[i],
                'low_range': jnp.asarray(ranges[i][0]),
                'high_range': jnp.asarray(ranges[i][1]),
                'sample_func':  sample_funcs[i],
                'logpdf_func': logpdf_funcs[i],
                'dist_type': dist_types[i].lower()
            }
        return prior_dict

    def sample(self, rng_key) -> dict[str, jnp.ndarray]:
        init_keys = jax.random.split(rng_key, self.ndim)
        params = {}
        for rng_key, name in zip(init_keys, self.prior_names):
            a = self.prior_dict[name]['low_range']
            b = self.prior_dict[name]['high_range']
            long_name = self.prior_dict[name]['simulator_kwarg']
            params[long_name] = self.prior_dict[name]['sample_func'](rng_key, a, b)
        return params

    def get_initial_live_samples(
            self, 
            rng_key: PRNGKey, 
            num_live: int
    ) -> dict[str, jnp.ndarray]:
        init_keys = jax.random.split(rng_key, num_live)
        particles = jax.vmap(self.sample)(init_keys)
        return particles

    def log_prob(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        log_prior = jnp.zeros(())
        for _, name in enumerate(self.prior_names): 
            long_name = self.prior_dict[name]['simulator_kwarg']
            x = params[long_name]
            a = self.prior_dict[name]['low_range']
            b = self.prior_dict[name]['high_range']
            log_prior += self.prior_dict[name]['logpdf_func'](x, a, b)
        return log_prior

    def log_prob_pray_its_ordered_correctly(self, params: jnp.ndarray) -> jnp.ndarray:
        log_prior = jnp.zeros(())
        for i, name in enumerate(self.prior_names): 
            x = params[..., i]
            a = self.prior_dict[name]['low_range']
            b = self.prior_dict[name]['high_range']
            log_prior += self.prior_dict[name]['logpdf_func'](x, a, b)
        return log_prior

    def get_adapter(self) -> PytreeAdapter:
        rng_key = jax.random.PRNGKey(0)
        example_theta = self.sample(rng_key)
        return PytreeAdapter(example_theta)
    
    # def add_prior(self,
    #         prior: 'UniformWrapper',
    #         short_name: str,
    #         simulator_kwarg: str,
    #         index: int
    # ) -> None:
    #     '''
    #     :param prior: Torch distribution, like UniformWrapper.
    #     :param index: Index at which to add the prior. E.g. specifying 0
    #         makes the prior the first in the prior list, 1 the 2nd, and so on...
    #     '''
    #     self.prior_dict[short_name] = {
    #         'simulator_kwarg': simulator_kwarg,
    #         'low_range': jnp.asarray(prior.low),
    #         'high_range': jnp.asarray(prior.high),
    #         'dist': prior
    #     }
    #     self.prior_names.insert(index, short_name)
    
    def write_prior_info(self, path):
        with open(path, "w") as f:
            f.write(f"Dimension: {self.ndim}\n")
            f.write("Variables:\n")
            for name in self.prior_names:
                low = self.prior_dict[name]['low_range']
                high = self.prior_dict[name]['high_range']
                kwarg = self.prior_dict[name]['simulator_kwarg']
                sample_func = self.prior_dict[name]['sample_func']
                dist_label = sample_func.__name__
                f.write(f"  {name} ({kwarg}): {dist_label}[{low}, {high}]\n")
    
class DipolePriorJax(JaxPrior):
    def __init__(self,
            mean_count_range: list[float] = [0.,   100. ],
            speed_range:      list[float] = [0.,   5.   ],
            longitude_range:  list[float] = [0.,   360. ],
            latitude_range:   list[float] = [-90., 90.  ],
    ) -> None:
        super().__init__()
        ranges = [mean_count_range, speed_range, longitude_range, latitude_range]
        self._prior_names = ['N', 'D', 'phi', 'theta']
        kwargs = [
            'n_initial_samples', 'observer_speed',
            'dipole_longitude', 'dipole_latitude'
        ]
        unif_func = lambda key, a, b: jax.random.uniform(key, minval=a, maxval=b)
        polar_func = lambda key, a, b: sample_polar_jax(key, minval=a, maxval=b)
        unif_logpdf = lambda x, a, b: jax.scipy.stats.uniform.logpdf(x, a, b - a)
        polar_logpdf = lambda x, a, b: polar_logpdf_jax(x, minval=a, maxval=b)

        sample_func = 3 * [unif_func] + [polar_func]
        logpdf_func = 3 * [unif_logpdf] + [polar_logpdf]
        dist_types = 3 * ['uniform'] + ['polar']

        self._prior_dict = self._construct_prior_dict(
            self._prior_names,
            kwargs,
            ranges,
            sample_func,
            logpdf_func,
            dist_types
        )
    
    @property
    def prior_names(self) -> list[str]: 
        return self._prior_names

    @property
    def prior_dict(self) -> dict[str, dict]:
        return self._prior_dict
