import torch
from torch.types import Tensor
from sbi.utils import BoxUniform
from abc import ABC, abstractmethod

class PriorMixin(ABC):
    @property
    @abstractmethod
    def prior_list(self) -> list:
        pass
    
    def sample(self, sample_size=torch.Size([])) -> Tensor:
        samples = []
        for prior in self.prior_list:
            samples.append( prior.sample(sample_size).flatten() )
        return torch.stack(samples, dim=1)
    
    def to(self, device: str) -> None:
        for prior in self.prior_list:
            prior.to(device)
        
        # these are estimated by sbi later and cause FUCKED problems if they
        # aren't on the right device
        if hasattr(self, 'mean'):
            self.mean = self.mean.to(device)
            self.variance = self.variance.to(device)
        
        self.device = device

    def log_prob(self, sample_values: Tensor) -> Tensor:
        n_samples = sample_values.shape[0]
        log_prob = torch.zeros(n_samples)
        for i, prior in enumerate(self.prior_list):
            log_prob += prior.log_prob(sample_values[:, i])
        return torch.as_tensor(log_prob)
    
    def add_prior(self,
            prior: BoxUniform,
            index: int
    ) -> None:
        '''
        :param prior: Torch distribution, like BoxUniform.
        :param index: Index at which to add the prior. E.g. specifying 0
            makes the prior the first in the prior, 1 the 2nd, and so on...
        '''
        assert hasattr(self, '_prior_list')
        self.prior_list.insert(index, prior)
        self.dimension += 1

        self.low_ranges.insert(index, float(prior.low))
        self.high_ranges.insert(index, float(prior.high))
    
    def write_out(self, path):
        with open(path, "w") as f:
            f.write(f"Dimension: {self.dimension}\n")
            f.write("Ranges:\n")
            for i, (low, high) in enumerate(zip(self.low_ranges, self.high_ranges)):
                f.write(f"  Param {i}: [{low}, {high}]\n")
    
    @property
    @abstractmethod
    def low_ranges(self) -> list:
        pass

    @property
    @abstractmethod
    def high_ranges(self) -> list:
        pass

class DipolePrior(PriorMixin):
    def __init__(self,
            mean_count_range: list[float] = [0, 100],
            amplitude_range: list[float]  = [0.0, 0.1],
            longitude_range: list[float]  = [0, 2*torch.pi],
            latitude_range: list[float]   = [0, torch.pi],
            return_numpy: bool = False
    ) -> None:
        self.return_numpy = return_numpy

        self._low_ranges = [
            rngs[0] for rngs in
            [mean_count_range, amplitude_range, longitude_range, latitude_range]
        ]
        self._high_ranges = [
            rngs[1] for rngs in
            [mean_count_range, amplitude_range, longitude_range, latitude_range]
        ]
        
        self.mean_count_dist = BoxUniform(
            low=mean_count_range[0]*torch.ones(1),
            high=mean_count_range[1]*torch.ones(1)
        )
        self.amplitude_dist = BoxUniform(
            low=amplitude_range[0]*torch.ones(1),
            high=amplitude_range[1]*torch.ones(1)
        )
        self.longitude_dist = BoxUniform(
            low=longitude_range[0]*torch.ones(1),
            high=longitude_range[1]*torch.ones(1)
        )
        self.latitude_dist = PolarPrior(
            theta_low=latitude_range[0]*torch.ones(1),
            theta_high=latitude_range[1]*torch.ones(1)
        )
        
        self.dimension = 4
        self._prior_list = [
            self.mean_count_dist, self.amplitude_dist, self.longitude_dist,
            self.latitude_dist
        ]
    
    @property
    def prior_list(self):
        return self._prior_list

    @property
    def low_ranges(self) -> list:
        return self._low_ranges

    @property
    def high_ranges(self) -> list:
        return self._high_ranges

class PolarPrior:
    def __init__(self,
            theta_low, theta_high, return_numpy: bool = False
        ) -> None:
        self.return_numpy = return_numpy
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.ndim = len(theta_low)
        self.device = 'cpu'

    def sample(self, sample_shape=torch.Size([])):
        '''
        :param sample_shape: shape of output in batchwise format (Nsamp, Ndim)
        '''
        samples = self.generate_polar(sample_shape)
        return samples.numpy() if self.return_numpy else samples

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
            return unif_theta[:, None]
        else:
            return unif_theta

    def polar_logpdf(self, theta):
        '''Probably density of polar angle evaluated at theta for polar angles
        between [theta_low, theta_high]'''
        p_theta = - torch.sin(theta) / (
            torch.cos(self.theta_high) - torch.cos(self.theta_low)
        )
        return torch.log(p_theta)

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = self.polar_logpdf(values)
        return log_probs.numpy() if self.return_numpy else log_probs
    
    def to(self, device: str) -> None:
        self.theta_low = self.theta_low.to(device)
        self.theta_high = self.theta_high.to(device)
        self.device = device