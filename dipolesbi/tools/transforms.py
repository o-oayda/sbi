from typing import Tuple
import torch
from torch import Tensor
from nflows.transforms.base import Transform


class LogAffineTransform(Transform):
    def __init__(self, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x, context=None) -> Tuple[Tensor, Tensor]: # type: ignore
        z = (torch.log1p(x) - self.mu) / self.sigma
        logabsdet = (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)
        return z, logabsdet

    def inverse(self, z, context=None) -> Tuple[Tensor, Tensor]: # type: ignore
        x = torch.expm1(z * self.sigma + self.mu)
        logabsdet = -(-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)
        return x, logabsdet

    def log_abs_det_jacobian(self, x, z):
        return (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)

class AnscombeTransform(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, x, context=None):
        z = 2 * torch.sqrt(x + 0.375)
        logabsdet = self.log_abs_det_jacobian(x, z)
        return z, logabsdet

    def inverse(self, z, context=None):
        x = z**2 / 4 - 0.375
        logabsdet = -self.log_abs_det_jacobian(x, z)
        return x, logabsdet

    def log_abs_det_jacobian(self, x, z):
        return - 0.5 * torch.log(x + 0.375).sum(dim=-1)
