from torch.types import Tensor
import torch
from typing import Callable

def parse_noise_model(flux_percentage_noise: float | str) -> float | Callable:
    if type(flux_percentage_noise) is float:
        return flux_percentage_noise
    elif type(flux_percentage_noise) is str:
        noise_str_to_model = {'ecliptic': ecliptic_noise}
        return noise_str_to_model[flux_percentage_noise]

def ecliptic_noise(
        fluxes: Tensor,
        declinations: Tensor,
        max_percentage_noise: float = 0.2,
        positive_declination_slope: float = 0.002,
        negative_declination_slope: float = 0.001
    ) -> Tensor:
    percentage_error = torch.zeros(len(fluxes))
    gtr_zero = declinations >= 0
    lss_zero = declinations < 0
    percentage_error[gtr_zero] = (
        max_percentage_noise
        - positive_declination_slope * declinations[gtr_zero]
    )
    percentage_error[lss_zero] = (
        max_percentage_noise
        - negative_declination_slope * torch.abs(declinations[lss_zero])
    )
    noise = torch.normal(
        mean=0,
        std=fluxes * percentage_error
    )
    return noise