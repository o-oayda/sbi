import torch
from torch.types import Tensor
from torch.distributions import Uniform
from abc import ABC, abstractmethod
from typing import Literal
from dipolesbi.tools.utils import sample_polar, sample_unif


class Prior(ABC):
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
    def low_ranges(self) -> Tensor:
        low_ranges = torch.empty(self.ndim)
        for i, name in enumerate(self.prior_names):
            low_ranges[i] = self.prior_dict[name]['low_range']
        return low_ranges 

    @property
    def high_ranges(self) -> Tensor:
        high_ranges = torch.empty(self.ndim)
        for i, name in enumerate(self.prior_names):
            high_ranges[i] = self.prior_dict[name]['high_range']
        return high_ranges 

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
            distributions: list
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
                'low_range': ranges[i][0],
                'high_range': ranges[i][1],
                'dist': distributions[i](
                    ranges[i][0] * torch.ones(1),
                    ranges[i][1] * torch.ones(1)
                )
            }
        return prior_dict

    def sample(self, sample_size=torch.Size([])) -> Tensor:
        samples = torch.empty(*sample_size, self.ndim, device=self.device) 
        for i, name in enumerate(self.prior_names):
            samples[..., i] = (
                self.prior_dict[name]['dist'].sample(sample_size).flatten()
            )
        return samples
    
    def transform(self, uniform_deviates: Tensor) -> Tensor:
        native_samples = torch.empty_like(uniform_deviates)
        for i, name in enumerate(self.prior_names):
            native_samples[..., i] = (
                self.prior_dict[name]['dist'].transform(uniform_deviates[..., i])
            )
        return native_samples

    def to(self, device: str) -> None:
        for name in self.prior_names: 
            self.prior_dict[name]['dist'].to(device)
        
        # these are estimated by sbi later and cause FUCKED problems if they
        # aren't on the right device
        # self.mean = self.mean.to(device)
        # self.variance = self.variance.to(device)
        
        self.device = device

    def log_prob(self, sample_values: Tensor) -> Tensor:
        n_samples = sample_values.shape[0]
        log_prob = torch.zeros(n_samples)
        for i, name in enumerate(self.prior_names): 
            log_prob += self.prior_dict[name]['dist'].log_prob(sample_values[:, i])
        return torch.as_tensor(log_prob)

    @property
    def mean(self) -> Tensor:
        mean_vals = torch.nan * torch.empty(self.ndim, device=self.device)
        for i, name in enumerate(self.prior_names):
            mean_vals[i] = self.prior_dict[name]['dist'].mean()
        return mean_vals

    @property
    def variance(self) -> Tensor:
        variance_vals = torch.nan * torch.empty(self.ndim, device=self.device)
        for i, name in enumerate(self.prior_names):
            variance_vals[i] = self.prior_dict[name]['dist'].variance()
        return variance_vals
    
    def add_prior(self,
            prior: 'UniformWrapper',
            short_name: str,
            simulator_kwarg: str,
            index: int
    ) -> None:
        '''
        :param prior: Torch distribution, like UniformWrapper.
        :param index: Index at which to add the prior. E.g. specifying 0
            makes the prior the first in the prior list, 1 the 2nd, and so on...
        '''
        self.prior_dict[short_name] = {
            'simulator_kwarg': simulator_kwarg,
            'low_range': float(prior.low),
            'high_range': float(prior.high),
            'dist': prior
        }
        self.prior_names.insert(index, short_name)
    
    def write_prior_info(self, path):
        with open(path, "w") as f:
            f.write(f"Dimension: {self.ndim}\n")
            f.write("Variables:\n")
            for name in self.prior_names:
                low = self.prior_dict[name]['low_range']
                high = self.prior_dict[name]['high_range']
                kwarg = self.prior_dict[name]['simulator_kwarg']
                dist = self.prior_dict[name]['dist']
                dist_label = self._dist_to_label(dist.__class__.__name__)
                f.write(f"  {name} ({kwarg}): {dist_label}[{low}, {high}]\n")

    def _dist_to_label(self, dist: Literal['UniformWrapper', 'PolarPrior']):
        mapping = {
            'UniformWrapper': 'Uniform',
            'PolarPrior': 'Polar'
        }
        return mapping[dist]
    
class DipolePrior(Prior):
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
        # do not use BoxUniform otherwise sbi fucks you in the ass with the log prob
        distributions = [UniformWrapper, UniformWrapper, UniformWrapper, PolarPrior]
        self._prior_dict = self._construct_prior_dict(
            self._prior_names,
            kwargs,
            ranges,
            distributions
        )
    
    @property
    def prior_names(self) -> list[str]: 
        return self._prior_names

    @property
    def prior_dict(self) -> dict[str, dict]:
        return self._prior_dict

class UniformWrapper:
    def __init__(self, low: Tensor, high: Tensor, device: str = 'cpu') -> None:
        self.low = low
        self.high = high
        self.distribution = Uniform(low, high)
        self.device = device

    def to(self, device: str) -> None:
        self.device = device
        self.distribution = Uniform(
            self.low.to(self.device), 
            self.high.to(self.device)
        )

    def sample(self, sample_shape=torch.Size([])) -> Tensor:
        return self.distribution.sample(sample_shape)
    
    def transform(self, uniform_deviates: Tensor) -> Tensor:
        return sample_unif(uniform_deviates, low_high=(self.low, self.high))

    def log_prob(self, values: Tensor) -> Tensor:
        return self.distribution.log_prob(values)

    def mean(self) -> Tensor:
        return self.distribution.mean

    def variance(self) -> Tensor:
        return self.distribution.variance

class PolarPrior:
    def __init__(self,
         theta_low: Tensor, theta_high: Tensor
    ) -> None:
        # note that high and low are switched intentionally when going
        # from latitude and longitude in deg to colat and lon in rad
        self.theta_low = torch.deg2rad(90. - theta_high)
        self.theta_high = torch.deg2rad(90. - theta_low)
        self.ndim = len(theta_low)
        self.device = 'cpu'

    def sample(self, sample_shape=torch.Size([])):
        '''
        :param sample_shape: shape of output in batchwise format (Nsamp, Ndim)
        '''
        samples = self.generate_polar(sample_shape)
        return samples

    def generate_polar(self, shape: tuple):
        '''
        :param shape: shape of output in batchwise format (Nsamp, Ndim)
        :returns: uniform deviate on latitudinal (polar) angles

        - theta ~ acos(u * (cos theta_high - cos theta_low) + cos theta_low)
        - theta in [theta_low, theta_high], subdomain of [0, pi]
        '''
        # assert shape[-1] == 1
        u = torch.rand(shape, device=self.device)
        unif_theta = torch.arccos(
            torch.cos(self.theta_low)
            + u * (torch.cos(self.theta_high) - torch.cos(self.theta_low))
        )
        if len(shape) == 1:
            return 90. - torch.rad2deg(unif_theta[:, None])
        else:
            return 90. - torch.rad2deg(unif_theta)

    def transform(self, uniform_deviates: Tensor) -> Tensor:
        return 90. - torch.rad2deg(
            sample_polar(
                uniform_deviates, 
                low_high=(self.theta_low, self.theta_high)
            )
        )

    def polar_logpdf(self, theta):
        '''Probably density of polar angle evaluated at theta for polar angles
        between [theta_low, theta_high].

        :param theta: Polar angle in radians (colatitude).
        '''
        p_theta = - torch.sin(theta) / (
            torch.cos(self.theta_high) - torch.cos(self.theta_low)
        )
        return torch.log(p_theta)

    def log_prob(self, values):
        '''
        Introduce factor of pi/180 at the end to map P(theta_rad) to P(theta_deg).
        '''
        values = torch.deg2rad(90. - values)
        log_probs = self.polar_logpdf(values)
        return log_probs + torch.log(torch.as_tensor(torch.pi / 180.))
    
    def to(self, device: str) -> None:
        self.theta_low = self.theta_low.to(device)
        self.theta_high = self.theta_high.to(device)
        self.device = device

    def mean(self) -> Tensor:
        th = self.theta_high
        tl = self.theta_low
        cos_th = torch.cos(self.theta_high)
        cos_tl = torch.cos(self.theta_low)
        sin_th = torch.sin(self.theta_high)
        sin_tl = torch.sin(self.theta_low)

        prefactor = - 1 / (cos_th - cos_tl)
        postfactor = sin_th - th * cos_th - sin_tl + tl * cos_tl
        return 90. - torch.rad2deg(prefactor * postfactor)

    def variance(self) -> Tensor:
        th = self.theta_high
        tl = self.theta_low
        cos_th = torch.cos(self.theta_high)
        cos_tl = torch.cos(self.theta_low)
        sin_th = torch.sin(self.theta_high)
        sin_tl = torch.sin(self.theta_low)
        mean_deg = self.mean()
        mean_rad = torch.deg2rad(90. - mean_deg)

        theta_sq_prefactor = - 1 / (cos_th - cos_tl)
        theta_sq_postfactor = (
            2 * th * sin_th + (2 - th**2) * cos_th
          - 2 * tl * sin_tl - (2 - tl**2) * cos_tl
        )
        expectation_theta_sq = theta_sq_prefactor * theta_sq_postfactor
        variance = expectation_theta_sq - mean_rad**2
        variance_deg = torch.rad2deg(torch.sqrt(variance))**2

        return variance_deg
