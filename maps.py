import numpy as np
from utils import sph2cart
import healpy as hp
import torch
from torch import poisson
from torch.types import Tensor

class Mask:
    def __init__(self, nside: int = 32):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.all_pixel_indices = set(np.arange(self.npix))

    def equator_mask(self, mask_angle: float) -> list:
        south_pole_vec = hp.ang2vec(0, -90, lonlat=True)
        north_pole_vec = hp.ang2vec(0, 90, lonlat=True)
        north_pole_indices = hp.query_disc(
            self.nside, north_pole_vec, radius=np.deg2rad(90 - mask_angle)
        )
        south_pole_indices = hp.query_disc(
            self.nside, south_pole_vec, radius=np.deg2rad(90 - mask_angle)
        )
        masked_pixel_indices = (
            self.all_pixel_indices
            - set([*north_pole_indices, *south_pole_indices])
        )
        return list(masked_pixel_indices)

class SkyMap:
    def __init__(self, nside: int = 32, device: str = 'cpu'):
        self.nside = nside
        self.device = device
        self.mask = Mask(self.nside)
        self.mask_map = torch.zeros(self.mask.npix)

    def generate_dipole(self, Theta: Tensor) -> None:
        poisson_mean = self.dipole_signal(Theta)
        self._density_map = poisson(poisson_mean)

    def dipole_signal(self, Theta: Tensor) -> Tensor:
        if Theta.shape == (4,):
            Theta = Theta.reshape(1,4)

        n_batches = Theta.shape[0]
        pixel_indices = torch.arange(hp.nside2npix(self.nside))
        pixel_vectors = torch.as_tensor(
            torch.stack(
                hp.pix2vec(self.nside, pixel_indices, nest=True)
            ),
            device=self.device
        ).to(torch.float32)
        mean_count = Theta[:, 0]
        dipole_amplitude = Theta[:, 1]
        dipole_longitude = Theta[:, 2]
        dipole_latitude = Theta[:, 3]
        dipole_vector = dipole_amplitude.reshape((n_batches,1)) * sph2cart(
            (dipole_latitude, dipole_longitude),
            device=self.device
        )
        poisson_mean = mean_count.reshape((n_batches,1)) * (
            1 + torch.einsum('ij,jk', dipole_vector, pixel_vectors)
        )
        if n_batches == 1:
            return poisson_mean.flatten()
        else:
            return poisson_mean

    def mask_pixels(self, fill_value = None, **kwargs) -> None:
        self.kwarg_to_mask = {'equator_mask': self.mask.equator_mask}
        for key, val in kwargs.items():
            masked_pixel_indices = self.kwarg_to_mask[key](val)
            self.mask_map[masked_pixel_indices] = 1

        if fill_value == None:
            self.fill_value = torch.nan
        else:
            self.fill_value = fill_value

    @property
    def density_map(self):
        out = self._density_map
        out[self.mask_map == 1] = self.fill_value
        return out

    def batch_simulator(self,
            proposal_distribution,
            n_samples: int,
            mask_fill_value = None,
            **mask_kwargs
    ) -> tuple[Tensor]:
        Theta = proposal_distribution.sample((n_samples,))
        poisson_mean = self.dipole_signal(Theta)
        self.batch_density_maps = poisson(poisson_mean)

        if mask_fill_value == None:
            fill_value = torch.nan
        else:
            fill_value = mask_fill_value

        self.mask_pixels(**mask_kwargs)
        self.batch_mask_maps = self.mask_map.repeat((n_samples, 1))
        self.batch_mask_maps[self.batch_mask_maps == 1] = fill_value

        return (Theta, self.batch_density_maps)