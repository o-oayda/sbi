import torch
from sbi.utils import BoxUniform

class DipolePrior:
    def __init__(self,
            mean_count_range: list[float] = [0, 100],
            amplitude_range: list[float] = [0.0, 0.1],
            longitude_range: list[float] = [0, 2*torch.pi],
            latitude_range: list[float] = [0, torch.pi],
            return_numpy: bool = False
    ) -> None:
        self.return_numpy = return_numpy

        self.mean_count_range = mean_count_range
        self.amplitude_range = amplitude_range
        self.longitude_range = longitude_range
        self.latitude_range = latitude_range
        
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
    
    def sample(self, sample_size=torch.Size([])):
        count_samples = self.mean_count_dist.sample(sample_size).flatten()
        amplitude_samples = self.amplitude_dist.sample(sample_size).flatten()
        longitude_samples = self.longitude_dist.sample(sample_size).flatten()
        latitude_samples = self.latitude_dist.sample(sample_size).flatten()
        return torch.stack(
            [
                count_samples, amplitude_samples,
                longitude_samples, latitude_samples
            ],
            dim=1
        )
    
    def mean_count_log_prob(self, mean_count):
        return self.mean_count_dist.log_prob(mean_count)
    
    def amplitude_log_prob(self, amplitude):
        return self.amplitude_dist.log_prob(amplitude)
    
    def longitude_log_prob(self, longitude):
        return self.longitude_dist.log_prob(longitude)
    
    def latitude_log_prob(self, latitude):
        return self.latitude_dist.polar_logpdf(latitude)

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = (
             self.mean_count_log_prob(values[:, 0])
           + self.amplitude_log_prob (values[:, 1])
           + self.longitude_log_prob (values[:, 2])
           + self.latitude_log_prob  (values[:, 3])
        )
        return log_probs.numpy() if self.return_numpy else log_probs
    
    def to(self, device: str) -> None:
        self.mean_count_dist.to(device)
        self.amplitude_dist.to(device)
        self.longitude_dist.to(device)
        self.latitude_dist.to(device)
        self.device = device

    def get_low_ranges(self) -> list:
        return [
            self.mean_count_range[0], self.amplitude_range[0],
            self.longitude_range[0], self.latitude_range[0]   
        ]

    def get_high_ranges(self) -> list:
        return [
            self.mean_count_range[1], self.amplitude_range[1],
            self.longitude_range[1], self.latitude_range[1]   
        ]

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