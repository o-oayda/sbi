import numpy as np
from numpy.typing import NDArray
from dipolesbi.tools.points import (
    sample_points_with_flux, boost_points_with_flux, flux_cut,
    add_noise_to_fluxes
)
import healpy as hp
import torch
from torch import poisson
from torch.types import Tensor
from dipolesbi.tools.utils import (
    check_vectorised_input, spherical_to_cartesian, omega_to_theta,
    equatorial_to_ecliptic
)
from dipolesbi.tools.physics import ellis_baldwin_amplitude
from typing import Literal, Callable
from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_simulator,
)
from dipolesbi.tools.noise_models import parse_noise_model
from dipolesbi.tools.noise_models import ecliptic_noise
from astropy.coordinates import SkyCoord
import astropy.units as u

class Mask:
    def __init__(self, nside: int = 32):
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.all_pixel_indices = set(np.arange(self.npix))

    def equator_mask(self, mask_angle: float) -> list:
        south_pole_vec = hp.ang2vec(0, -90, lonlat=True)
        north_pole_vec = hp.ang2vec(0, 90, lonlat=True)
        north_pole_indices = hp.query_disc(
            self.nside, north_pole_vec, radius=np.deg2rad(90 - mask_angle),
            nest=True
        )
        south_pole_indices = hp.query_disc(
            self.nside, south_pole_vec, radius=np.deg2rad(90 - mask_angle),
            nest=True
        )
        masked_pixel_indices = (
            self.all_pixel_indices
            - set([*north_pole_indices, *south_pole_indices])
        )
        return list(masked_pixel_indices)
    
    def catwise_mask(self) -> list:
        '''
        Return CatWISE2020 mask used in Secrest et al. (2021) in Galactic
        coordinates, with `nside=64` and in nest ordering.
        '''
        galactic_mask = hp.reorder(
            np.load('dipolesbi/catwise/CatWISE_Mask_nside64.npy'),
            r2n=True
        )
        masked_pixel_indices = list(np.where(galactic_mask == 0)[0])
        return masked_pixel_indices
    
    def north_ecliptic_mask(self) -> list:
        ecl_north_pole = SkyCoord(
            lon=0 * u.deg,  # type: ignore
            lat=90 * u.deg, # type: ignore
            frame='geocentrictrueecliptic'
        )
        gal_north_pole = ecl_north_pole.transform_to('galactic')
        pole_colatitude = np.deg2rad(90 - gal_north_pole.b.deg)  # type: ignore
        pole_longitude = np.deg2rad(gal_north_pole.l.deg)        # type: ignore
        vec_north_ecl_gal = hp.ang2vec(pole_colatitude, pole_longitude)

        north_pole_disc_pixels = hp.query_disc(
            nside=self.nside,
            vec=vec_north_ecl_gal,
            radius=np.deg2rad(5), # hardcoded 5 degrees, see error_sampling_gp.py
            nest=True
        )
        return list(north_pole_disc_pixels)

class SkyMap:
    def __init__(self, nside: int = 32, device: str = 'cpu'):
        self.nside = nside
        self.device = device
        self.mask = Mask(self.nside)
        self.mask_map = torch.zeros(self.mask.npix)
        self.nest = True

    def configure(self,
            dipole_method: Literal['base', 'poisson'],
            dipole_hyperparameters: dict = {},
            mask_kwargs: dict | None = None
        ) -> None:
        '''
        Configures parameters for the batch simulator method.
        '''
        self.dipole_method = dipole_method
        self.dipole_hyperparameters = dipole_hyperparameters
        if mask_kwargs is None:
            self.mask_kwargs = {'mask_fill_value': None}
        else:
            self.mask_kwargs = mask_kwargs

    def generate_dipole(self, Theta: Tensor) -> None:
        poisson_mean = self.dipole_signal(Theta)
        self._density_map = poisson(poisson_mean)

    def dipole_signal(self, Theta: Tensor) -> Tensor:
        Theta = check_vectorised_input(Theta, ndim=4)

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
        dipole_vector = dipole_amplitude.reshape((n_batches,1)) * spherical_to_cartesian(
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

    def generate_dipole_from_base(self,
            observer_direction: tuple[float, float],
            n_initial_points: int,
            observer_speed: float = 0.00123,
            luminosity_function_index: int = 2,
            mean_spectral_index: float = 0.8,
            sigma_spectral_index: float = 0.5,
            flux_percentage_noise: float | str = 0.1,
            minimum_flux_cut: float = 5,
            noise_model_kwargs: dict = {}
        ) -> Tensor:
        '''
        :param observer_direction: Direction of observer's motion in spherical
            coordinates in radians (phi, theta).
        '''
        dipole_longitude_rad, dipole_colatitude_rad = torch.as_tensor(
            observer_direction
        )
        
        dipole_longitude = torch.rad2deg(dipole_longitude_rad)
        dipole_latitude = torch.rad2deg( torch.pi / 2 - dipole_colatitude_rad )

        longitudes_deg, latitudes_deg,\
        rest_fluxes, spectral_indices = sample_points_with_flux(
            n_initial_points=n_initial_points,
            luminosity_function_index=luminosity_function_index,
            mean_spectral_index=mean_spectral_index,
            sigma_spectral_index=sigma_spectral_index
        )

        self.boosted_longitudes_deg, self.boosted_latitudes_deg,\
        boosted_fluxes = boost_points_with_flux(
            longitudes_deg=longitudes_deg,
            latitudes_deg=latitudes_deg,
            rest_fluxes=rest_fluxes,
            spectral_indices=spectral_indices,
            observer_direction=(dipole_longitude, dipole_latitude),
            observer_speed=observer_speed
        )

        noise_model = parse_noise_model(flux_percentage_noise)
        noise_parameter = self.get_needed_noise_parameter(noise_model)
        boosted_fluxes = add_noise_to_fluxes(
            fluxes=boosted_fluxes,
            noise_model=noise_model,
            noise_scaling_parameter=noise_parameter,
            **noise_model_kwargs
        )

        cut_boosted_longitudes_deg, cut_boosted_latitudes_deg,\
        cut_boosted_fluxes = flux_cut(
            minimum_flux=minimum_flux_cut,
            longitudes=self.boosted_longitudes_deg,
            latitudes=self.boosted_latitudes_deg,
            fluxes=boosted_fluxes
        )

        self._density_map = self.make_density_map(
            cut_boosted_longitudes_deg, cut_boosted_latitudes_deg
        )
        self.expected_amplitude = ellis_baldwin_amplitude(
            observer_speed=observer_speed,
            mean_spectral_index=mean_spectral_index,
            luminosity_function_slope=luminosity_function_index
        )
        return self._density_map

    def get_needed_noise_parameter(self, noise_model: Callable) -> Tensor:
        if type(noise_model) is float:
            return None
        else:
            model_to_parameter = {
                ecliptic_noise: self.get_ecliptic_latitudes()
            }
            return model_to_parameter[noise_model]
    
    def get_ecliptic_latitudes(self) -> Tensor:
        _, boosted_ecliptic_latitudes_deg = equatorial_to_ecliptic(
            ra=self.boosted_longitudes_deg.numpy(),
            dec=self.boosted_latitudes_deg.numpy(),
            output_unit='degrees'
        )
        return torch.as_tensor(
            boosted_ecliptic_latitudes_deg,
            dtype=torch.float32
        )

    def make_density_map(self, longitudes: Tensor, latitudes: Tensor) -> None:
        source_indices = hp.ang2pix(
            self.nside, longitudes, latitudes, lonlat=True, nest=self.nest
        )
        return torch.bincount(
            source_indices, minlength=hp.nside2npix(self.nside)
        )

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
            prior_returns_numpy: bool,
            n_samples: int,
            n_workers: int = 32
    ) -> tuple[Tensor, Tensor]:
        self.n_samples = n_samples

        if self.dipole_method == 'poisson':
            Theta, self.batch_density_maps = self.poisson_batches(
                proposal_distribution=proposal_distribution,
                n_samples=n_samples
            )
        
        elif self.dipole_method == 'base':
            
            def simulator(Theta):
                N, v = Theta[0], Theta[1]
                phi, theta = Theta[2], Theta[3]
                int_N = N.to(torch.int32).item()

                density_map = self.generate_dipole_from_base(
                    observer_direction=(phi, theta),
                    n_initial_points=int_N,
                    observer_speed=v,
                    **self.dipole_hyperparameters
                )
                return density_map
            
            simulator = process_simulator(
                simulator, proposal_distribution, prior_returns_numpy
            )
            check_sbi_inputs(simulator=simulator, prior=proposal_distribution)
            Theta, self.batch_density_maps = simulate_for_sbi(
                simulator=simulator,
                proposal=proposal_distribution,
                num_workers=n_workers,
                num_simulations=n_samples
            )
        else:
            raise Exception('Method not recognised.')
        
        self.batchwise_mask(**self.mask_kwargs)

        return Theta, self.batch_density_maps

    def batchwise_mask(self, mask_fill_value, **mask_kwargs) -> None:
        if mask_fill_value == None:
            fill_value = torch.nan
        else:
            fill_value = mask_fill_value

        self.mask_pixels(**mask_kwargs)
        self.batch_mask_maps = self.mask_map.repeat((self.n_samples, 1))
        self.batch_density_maps[self.batch_mask_maps == 1] = fill_value
    
    def poisson_batches(self,
            proposal_distribution,
            n_samples: int
    ) -> tuple[Tensor]:
        Theta = proposal_distribution.sample((n_samples,))
        poisson_mean = self.dipole_signal(Theta)
        batch_density_maps = poisson(poisson_mean)
        return (Theta, batch_density_maps)

def average_smooth_map(
        healpy_map: NDArray[np.floating],
        weights: NDArray[np.floating] | None = None, 
        angle_scale: float = 1.
    ) -> NDArray:
    '''
    Smooth a healpy map using a moving average.
    '''
    included_pixels = np.where(~np.isnan(healpy_map))[0]
    smoothed_map = np.nan * np.empty_like(healpy_map)
    nside = hp.get_nside(healpy_map)
    
    if weights is None:
        weights = np.ones_like(healpy_map)

    smoothing_radius = omega_to_theta(angle_scale)
    for p_index in included_pixels:
        vec = hp.pix2vec(nside, p_index, nest=True)
        disc = hp.query_disc(nside, vec, smoothing_radius, nest=True)
        smoothed_map[p_index] = np.nanmean(healpy_map[disc] * weights[disc])

    return smoothed_map
