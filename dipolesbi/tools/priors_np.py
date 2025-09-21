from abc import ABC, abstractmethod
from typing import Callable
import scipy
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.utils import polar_logpdf_np, sample_polar_np, sample_unif_np
import numpy as np
from numpy.typing import NDArray, DTypeLike


class NPPrior(ABC):
    def __init__(self, dtype: DTypeLike) -> None:
        self.dtype = dtype

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
    def low_ranges(self) -> NDArray:
        low_ranges = np.empty(self.ndim)
        for i, name in enumerate(self.prior_names):
            low_ranges[i] = self.prior_dict[name]['low_range']
        return low_ranges 

    @property
    def high_ranges(self) -> NDArray:
        high_ranges = np.empty(self.ndim)
        for i, name in enumerate(self.prior_names):
            high_ranges[i] = self.prior_dict[name]['high_range']
        return high_ranges 

    @property
    def simulator_kwargs(self) -> list[str]:
        return [
            self.prior_dict[name]['simulator_kwarg'] for name in self.prior_names
        ]

    def change_kwarg(self, param_short_name: str, new_kwarg: str) -> None:
        self.prior_dict[param_short_name]['simulator_kwarg'] = new_kwarg

    def _construct_prior_dict(
            self,
            short_names: list[str],            
            simulator_kwargs: list[str],
            ranges: list[list[float]],
            sample_funcs: list[Callable],
            logpdf_funcs: list[Callable],
            tform_funcs: list[Callable]
            
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
                'low_range': np.asarray(ranges[i][0], dtype=self.dtype),
                'high_range': np.asarray(ranges[i][1], dtype=self.dtype),
                'sample_func':  sample_funcs[i],
                'logpdf_func': logpdf_funcs[i],
                'tform_func': tform_funcs[i]
            }
        return prior_dict

    def sample(self, rng_key: NPKey, n_samples: int) -> dict[str, NDArray]:
        init_keys = rng_key.split(self.ndim)
        params = {}
        for rng_key, name in zip(init_keys, self.prior_names):
            a = self.prior_dict[name]['low_range']
            b = self.prior_dict[name]['high_range']
            long_name = self.prior_dict[name]['simulator_kwarg']
            params[long_name] = self.prior_dict[name]['sample_func'](
                rng_key, n_samples, a, b
            )
        return params

    def transform(self, unifs: dict[str, NDArray]) -> dict[str, NDArray]:
        transformed = {}
        for key, val in unifs.items():
            a = self.prior_dict[key]['low_range']
            b = self.prior_dict[key]['high_range']
            tform = self.prior_dict[key]['tform_func']
            transformed[key] = tform(val, a, b)
        return transformed

    def get_initial_live_samples(
            self, 
            rng_key: NPKey, 
            num_live: int
    ) -> dict[str, NDArray]:
        return self.sample(rng_key, num_live)

    def log_prob(self, params: dict[str, NDArray]) -> NDArray:
        n_batches = len(params[ next(iter(params)) ])
        log_prior = np.zeros((n_batches,))
        for _, name in enumerate(self.prior_names): 
            long_name = self.prior_dict[name]['simulator_kwarg']
            x = params[long_name]
            a = self.prior_dict[name]['low_range']
            b = self.prior_dict[name]['high_range']
            log_prior += self.prior_dict[name]['logpdf_func'](x, a, b)
        return log_prior
    
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
    
class DipolePriorNP(NPPrior):
    def __init__(self,
            mean_count_range: list[float] = [0.,   100. ],
            speed_range:      list[float] = [0.,   5.   ],
            longitude_range:  list[float] = [0.,   360. ],
            latitude_range:   list[float] = [-90., 90.  ],
            dtype: DTypeLike = np.float32
    ) -> None:
        super().__init__(dtype)
        ranges = [mean_count_range, speed_range, longitude_range, latitude_range]
        self._prior_names = ['N', 'D', 'phi', 'theta']
        kwargs = [
            'n_initial_samples', 'observer_speed',
            'dipole_longitude', 'dipole_latitude'
        ]
        unif_func = lambda key, n, a, b: key.uniform((n,), low=a, high=b)
        polar_func = lambda key, n, a, b: sample_polar_np(key, n_samples=n, low=a, high=b)
        unif_logpdf = lambda x, a, b: scipy.stats.uniform.logpdf(x, a, b - a)
        polar_logpdf = lambda x, a, b: polar_logpdf_np(x, low=a, high=b)
        unif_transform = lambda u, a, b: sample_unif_np(u, low=a, high=b)
        polar_transform = lambda u, a, b: sample_polar_np(u, low=a, high=b)

        sample_func = 3 * [unif_func] + [polar_func]
        logpdf_func = 3 * [unif_logpdf] + [polar_logpdf]
        tform_func  = 3 * [unif_transform] + [polar_transform]

        self._prior_dict = self._construct_prior_dict(
            self._prior_names,
            kwargs,
            ranges,
            sample_func,
            logpdf_func,
            tform_func
        )
    
    @property
    def prior_names(self) -> list[str]: 
        return self._prior_names

    @property
    def prior_dict(self) -> dict[str, dict]:
        return self._prior_dict

    def to_jax(self) -> DipolePriorJax:
        # this is shit
        jax_prior = DipolePriorJax(
            mean_count_range=[self.low_ranges[0], self.high_ranges[0]],
            speed_range=[self.low_ranges[1], self.high_ranges[1]],
            longitude_range=[self.low_ranges[2], self.high_ranges[2]],
            latitude_range=[self.low_ranges[3], self.high_ranges[3]],
        )
        np_kwargs = self.simulator_kwargs
        jax_kwargs = jax_prior.simulator_kwargs

        for i, short_name in enumerate(self.prior_names):
            if np_kwargs[i] != jax_kwargs[i]:
                jax_prior.change_kwarg(short_name, np_kwargs[i])

        return jax_prior
