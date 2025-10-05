from abc import ABC, abstractmethod
from typing import Callable, Optional
import scipy
import jax
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.utils import (
    polar_logpdf_np,
    polar_logpdf_jax,
    sample_polar_np,
    sample_polar_jax,
    sample_unif_np,
)
import numpy as np
from numpy.typing import NDArray, DTypeLike


class NPPrior(ABC):
    def __init__(self, dtype: DTypeLike) -> None:
        self.dtype = dtype

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(dtype={self.dtype}"]
        for name in self.prior_names:
            entry = self.prior_dict[name]
            low = entry['low_range']
            high = entry['high_range']
            kwarg = entry['simulator_kwarg']
            dist = entry.get('dist_type', 'unknown')
            lines.append(
                f"  {name}: kwarg='{kwarg}', dist='{dist}', range=[{low}, {high}]"
            )
        lines.append(")")
        return "\n".join(lines)

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
        low_ranges = np.empty(self.ndim, dtype=self.dtype)
        for i, name in enumerate(self.prior_names):
            low_ranges[i] = self.prior_dict[name]['low_range']
        return low_ranges 

    @property
    def high_ranges(self) -> NDArray:
        high_ranges = np.empty(self.ndim, dtype=self.dtype)
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
            tform_funcs: list[Callable],
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
                'low_range': np.asarray(ranges[i][0], dtype=self.dtype),
                'high_range': np.asarray(ranges[i][1], dtype=self.dtype),
                'sample_func':  sample_funcs[i],
                'logpdf_func': logpdf_funcs[i],
                'tform_func': tform_funcs[i],
                'dist_type': dist_types[i].lower()
            }
        return prior_dict

    def _np_distribution_funcs(self, dist_type: str) -> tuple[Callable, Callable, Callable]:
        dist = dist_type.lower()
        if dist == 'uniform':
            def sample(key: NPKey, n: int, a, b): # pyright: ignore[reportRedeclaration]
                return key.uniform((n,), low=a, high=b, dtype=self.dtype)

            def logpdf(x, a, b): # pyright: ignore[reportRedeclaration]
                return scipy.stats.uniform.logpdf(x, loc=a, scale=b - a)

            def transform(u, a, b):
                return sample_unif_np(u, low=a, high=b)

            return sample, logpdf, transform

        elif dist == 'polar':
            def sample(key: NPKey, n: int, a, b):
                return sample_polar_np(key, n_samples=n, low=a, high=b, dtype=self.dtype)

            def logpdf(x, a, b):
                return polar_logpdf_np(x, low=a, high=b)

            def transform(u, a, b):
                return sample_polar_np(u, low=a, high=b, dtype=self.dtype)  # type: ignore[arg-type]

            return sample, logpdf, transform

        raise ValueError(f"Unknown distribution type '{dist_type}'.")

    def _jax_distribution_funcs(self, dist_type: str) -> tuple[Callable, Callable]:
        dist = dist_type.lower()
        if dist == 'uniform':
            def sample(key, a, b):
                return jax.random.uniform(key, minval=a, maxval=b)

            def logpdf(x, a, b):
                return jsp_stats.uniform.logpdf(x, loc=a, scale=b - a)

            return sample, logpdf

        if dist == 'polar':
            return sample_polar_jax, polar_logpdf_jax

        raise ValueError(f"Unknown distribution type '{dist_type}'.")

    def add_prior(
            self,
            short_name: str,
            simulator_kwarg: str,
            low: float,
            high: float,
            dist_type: str = 'uniform',
            index: Optional[int] = None
    ) -> None:
        if short_name in self.prior_dict:
            raise ValueError(f"Prior '{short_name}' already exists.")

        if index is None:
            index = len(self.prior_names)

        np_sample, np_logpdf, np_tform = self._np_distribution_funcs(dist_type)

        entry = {
            'simulator_kwarg': simulator_kwarg,
            'low_range': np.asarray(low, dtype=self.dtype),
            'high_range': np.asarray(high, dtype=self.dtype),
            'sample_func': np_sample,
            'logpdf_func': np_logpdf,
            'tform_func': np_tform,
            'dist_type': dist_type.lower()
        }

        self.prior_dict[short_name] = entry
        self.prior_names.insert(index, short_name)

    def remove_prior(self, short_name: str) -> None:
        if short_name not in self.prior_dict:
            raise ValueError(f"Prior '{short_name}' not found.")

        # Remove from the ordered name list first to keep downstream views consistent.
        self.prior_names.remove(short_name)
        del self.prior_dict[short_name]

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
        dist_types = ['uniform', 'uniform', 'uniform', 'polar']
        sample_func = []
        logpdf_func = []
        tform_func = []
        for dist in dist_types:
            sample, logpdf, transform = self._np_distribution_funcs(dist)
            sample_func.append(sample)
            logpdf_func.append(logpdf)
            tform_func.append(transform)

        self._prior_dict = self._construct_prior_dict(
            self._prior_names,
            kwargs,
            ranges,
            sample_func,
            logpdf_func,
            tform_func,
            dist_types
        )
    
    @property
    def prior_names(self) -> list[str]: 
        return self._prior_names

    @property
    def prior_dict(self) -> dict[str, dict]:
        return self._prior_dict

    def to_jax(self) -> DipolePriorJax:
        jax_prior = DipolePriorJax()
        jax_prior._prior_dict = {}
        jax_prior._prior_names = []

        for name in self.prior_names:
            entry = self.prior_dict[name]
            sample_func, logpdf_func = self._jax_distribution_funcs(entry['dist_type'])
            jax_prior._prior_dict[name] = {
                'simulator_kwarg': entry['simulator_kwarg'],
                'low_range': jnp.asarray(entry['low_range'], dtype=jnp.float32),
                'high_range': jnp.asarray(entry['high_range'], dtype=jnp.float32),
                'sample_func': sample_func,
                'logpdf_func': logpdf_func,
                'dist_type': entry['dist_type']
            }
            jax_prior._prior_names.append(name)

        return jax_prior
