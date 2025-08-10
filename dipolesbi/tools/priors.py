import torch
from torch.types import Tensor
from torch.distributions import Uniform
from abc import ABC, abstractmethod
from typing import Literal


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
    
    def to(self, device: str) -> None:
        for name in self.prior_names: 
            self.prior_dict[name]['dist'].to(device)
        
        # these are estimated by sbi later and cause FUCKED problems if they
        # aren't on the right device
        if hasattr(self, 'mean'):
            self.mean = self.mean.to(device)
            self.variance = self.variance.to(device)
        
        self.device = device

    def log_prob(self, sample_values: Tensor) -> Tensor:
        n_samples = sample_values.shape[0]
        log_prob = torch.zeros(n_samples)
        for i, name in enumerate(self.prior_names): 
            log_prob += self.prior_dict[name]['dist'].log_prob(sample_values[:, i])
        return torch.as_tensor(log_prob)
    
    def add_prior(self,
            prior: Uniform,
            short_name: str,
            simulator_kwarg: str,
            index: int
    ) -> None:
        '''
        :param prior: Torch distribution, like Uniform.
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

    def _dist_to_label(self, dist: Literal['Uniform', 'PolarPrior']):
        mapping = {
            'Uniform': 'Uniform',
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
        distributions = [Uniform, Uniform, Uniform, PolarPrior]
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

    def polar_logpdf(self, theta):
        '''Probably density of polar angle evaluated at theta for polar angles
        between [theta_low, theta_high]'''
        p_theta = - torch.sin(theta) / (
            torch.cos(self.theta_high) - torch.cos(self.theta_low)
        )
        return torch.log(p_theta)

    def log_prob(self, values):
        values = torch.deg2rad(90. - values)
        log_probs = self.polar_logpdf(values)
        return log_probs
    
    def to(self, device: str) -> None:
        self.theta_low = self.theta_low.to(device)
        self.theta_high = self.theta_high.to(device)
        self.device = device
