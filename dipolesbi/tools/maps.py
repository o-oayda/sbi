import typing
import jax
import numpy as np
from numpy.typing import DTypeLike, NDArray
import scipy as sp
from dipolesbi.tools.configs import SimpleDipoleMapConfig
from dipolesbi.tools.np_rngkey import NPKey
import healpy as hp
from dipolesbi.tools.np_rngkey import poisson
from dipolesbi.tools.utils import (
    jax_sph2cart, spherical_to_cartesian, omega_to_theta,
)
from dipolesbi.tools.healpix_helpers import downgrade_ignore_nan, downgrade_ignore_nan_jax
from typing import Optional
from astropy.coordinates import SkyCoord
import astropy.units as u
from jax import numpy as jnp


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

class SimpleDipoleMapJax:
    def __init__(
            self, 
            nside: int = 64, 
            reference_data: Optional[jnp.ndarray] = None,
            reference_mask: Optional[jnp.ndarray] = None,
            downscale_nside: Optional[int] = None
    ) -> None:
        self.nside = nside
        self.fiducial_amplitude = 0.005
        self.nest = True
        self.n_pix: int = hp.nside2npix(self.nside)
        self.pixel_indices = jnp.arange(self.n_pix)
        self.pixel_vectors = np.stack(
            hp.pix2vec(self.nside, self.pixel_indices, nest=True)
        )

        self.reference_data = (
            None if reference_data is None else jnp.asarray(reference_data)
        )
        self.reference_mask = (
            None if reference_mask is None else jnp.asarray(reference_mask, dtype=bool)
        )
        self.downscale_nside = downscale_nside

        if self.reference_data is not None:
            assert self.reference_mask is not None, (
                "A binary mask must also be supplied when supplying reference data."
            )
            assert self.reference_mask.ndim == 1, "reference_mask must be 1D."

            native_npix = self.n_pix
            assert self.reference_mask.shape[-1] == native_npix, (
                'reference_mask must be specified at the native resolution.'
            )

            assert self.reference_data.ndim == 1, "reference_data must be 1D."
            if self.downscale_nside is None:
                assert self.reference_data.shape[-1] == native_npix, (
                    'reference_data must match native resolution when no '
                    'downscaling is configured.'
                )
            else:
                coarse_npix = hp.nside2npix(self.downscale_nside)
                assert self.reference_data.shape[-1] == coarse_npix, (
                    'reference_data must match the downscaled resolution.'
                )

        if self.downscale_nside is not None:
            assert self.nside >= self.downscale_nside, (
                'downscale_nside must not exceed native nside.'
            )
            ratio = self.nside // self.downscale_nside
            assert (self.nside % self.downscale_nside) == 0 and (ratio & (ratio - 1)) == 0

    def generate_dipole(
            self,
            rng_key: typing.Any, # fix issues with jax when using batch_simulate 
            theta: dict[str, jnp.ndarray],
            make_poisson_draws: bool = True
    ) -> jnp.ndarray:
        poisson_mean = self.dipole_signal(**theta)
        if make_poisson_draws:
            return jax.random.poisson(rng_key, poisson_mean)
        else:
            return poisson_mean

    def dipole_signal(
            self, 
            mean_density: jnp.ndarray, 
            observer_speed: jnp.ndarray, 
            dipole_longitude: jnp.ndarray,
            dipole_latitude: jnp.ndarray
    ) -> jnp.ndarray:
        dipole_longitude_rad = jnp.deg2rad(dipole_longitude)
        dipole_colatitude_rad = jnp.pi / 2 - jnp.deg2rad(dipole_latitude)

        D_x, D_y, D_z = jax_sph2cart(dipole_longitude_rad, dipole_colatitude_rad)
        dipole_unit_vector = jnp.stack([D_x, D_y, D_z])
        dipole_amplitude = observer_speed * self.fiducial_amplitude
        dipole_vector = dipole_amplitude * dipole_unit_vector
        poisson_mean = mean_density * (
            1 + jnp.einsum('i,ij', dipole_vector, self.pixel_vectors)
        )
        return poisson_mean

    def log_likelihood(self, theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
        assert self.reference_data is not None
        dipole_signal = self.dipole_signal(**theta)
        logl = lambda k, mu: jnp.sum(jax.scipy.stats.poisson.logpmf(k=k, mu=mu))

        if self.reference_mask is None:
            assert self.reference_data.shape[0] == dipole_signal.shape[0], (
                'reference_data must match native resolution.'
            )
            return logl(self.reference_data, dipole_signal)

        fine_mask = self.reference_mask.astype(bool)
        # jax complaints about dynamic bool conversions
        # if not bool(jnp.any(fine_mask)):
        #     raise ValueError('reference_mask removes all pixels; cannot compute likelihood.')

        if self.downscale_nside:
            coarse_signal, coarse_mask = downgrade_ignore_nan_jax(
                dipole_signal,
                fine_mask,
                self.downscale_nside
            )

            dummy_map = jnp.where(fine_mask, jnp.zeros_like(dipole_signal), jnp.nan)
            _, coarse_mask_from_mask = downgrade_ignore_nan_jax(
                dummy_map,
                fine_mask,
                self.downscale_nside
            )

            # jax complaints about dynamic bool conversions
            # if not bool(jnp.all(coarse_mask == coarse_mask_from_mask)):
            #     raise ValueError('Mask differs across batches after downscaling.')

            assert self.reference_data.shape[0] == coarse_signal.shape[0], (
                'reference_data must match the downscaled resolution.'
            )

            effective_mask = coarse_mask
            observed_counts = self.reference_data
            model_counts = coarse_signal
        else:
            assert self.reference_data.shape[0] == dipole_signal.shape[0], (
                'reference_data must match native resolution.'
            )
            effective_mask = fine_mask
            observed_counts = self.reference_data
            model_counts = dipole_signal

        # Boolean masking is not JIT-friendly in JAX because the indices need to be
        # concrete. Instead, build dense arrays with safe filler values and gate the
        # contribution using the mask so the computation stays compatible with tracing.
        safe_counts = jnp.where(effective_mask, observed_counts, 0.0)
        safe_means = jnp.where(effective_mask, model_counts, 1.0)
        logpmf_vals = jax.scipy.stats.poisson.logpmf(k=safe_counts, mu=safe_means)
        return jnp.sum(jnp.where(effective_mask, logpmf_vals, 0.0))

class SimpleDipoleMap:
    def __init__(self, config: SimpleDipoleMapConfig) -> None:
        '''
        See SimpleDipoleMapConfig for details.
        '''
        self.config = config
        self.nside = self.config.nside
        self.dtype = self.config.dtype
        self.fiducial_amplitude = 0.005
        self.nest = True
        self.mask = Mask(nside=self.config.nside)
        self.masked_pixels = set()
        self._data_masked_val = self.config.is_masked_val
        self.downscale_nside = self.config.downscale_nside

        self.reference_data = self.config.reference_data
        self.reference_mask = self.config.reference_mask

        if self.reference_data is not None:
            assert self.reference_mask is not None, (
                "A binary mask must also be supplied when supplying reference data."
            )
            assert self.reference_mask.ndim == 1, "reference_mask must be 1D."

            native_npix = hp.nside2npix(self.nside)
            assert self.reference_mask.shape[-1] == native_npix, (
                'reference_mask must be specified at the native resolution.'
            )

            assert self.reference_data.ndim == 1, "reference_data must be 1D."
            if self.downscale_nside is None:
                assert self.reference_data.shape[-1] == native_npix, (
                    'reference_data must match the native resolution when '
                    'no downscaling is configured.'
                )
            else:
                coarse_npix = hp.nside2npix(self.downscale_nside)
                assert self.reference_data.shape[-1] == coarse_npix, (
                    'reference_data must match the downscaled resolution.'
                )

        if self.downscale_nside is not None:
            assert self.nside >= self.downscale_nside, (
                'downscale_nside must not exceed native nside.'
            )

            ratio = self.nside // self.downscale_nside
            assert ( # power of 2 check
                    (self.nside % self.downscale_nside) == 0
                and (ratio & (ratio - 1)) == 0
            )
    
    def equatorial_plane_mask(self, angle: float) -> None:
        self.masked_pixels |= set(self.mask.equator_mask(angle))

    def catwise_mask(self) -> None:
        assert self.nside == 64

        self.masked_pixels |= set(self.mask.catwise_mask())
        north_pole_pixels = self.mask.north_ecliptic_mask()
        self.masked_pixels.update(north_pole_pixels)

    def generate_dipole(self,
            rng_key: NPKey,
            theta: dict[str, NDArray],
            make_poisson_draws: bool = True
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        for key in theta.keys():
            if theta[key].shape == ():
                theta[key] = theta[key].reshape((1,))
        poisson_mean = self.dipole_signal(**theta)
        if make_poisson_draws:
            self._density_map = poisson(
                rng_key, 
                lam=poisson_mean, 
                shape=poisson_mean.shape
            ).astype(self.dtype)
        else:
            self._density_map = poisson_mean.astype(self.dtype)

        # if downscale is desired, transform after generating poisson deviates
        if self.downscale_nside:
            dmap, mask = self.dmap_and_mask
            return downgrade_ignore_nan(dmap, mask, self.downscale_nside)
        else:
            return self.dmap_and_mask

    @property
    def dmap_and_mask(self) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        out_map = self._density_map.copy()
        mask_map = np.ones_like(self._density_map, dtype=np.bool_)
        mask_map[:, list(self.masked_pixels)] = False
        out_map[~mask_map] = self._data_masked_val
        return out_map, mask_map

    def dipole_signal(
            self,
            mean_density: NDArray[np.floating],
            observer_speed: NDArray[np.floating],
            dipole_longitude: NDArray[np.floating],
            dipole_latitude: NDArray[np.floating],
    ) -> NDArray:
        n_batches = len(mean_density)
        pixel_indices = np.arange(hp.nside2npix(self.nside))
        pixel_vectors = np.stack(
            hp.pix2vec(self.nside, pixel_indices, nest=True)
        )

        mean_count = mean_density
        dipole_amplitude = observer_speed * self.fiducial_amplitude
        dipole_longitude_rad = np.deg2rad(dipole_longitude)
        dipole_colatitude_rad = np.pi / 2 - np.deg2rad(dipole_latitude)

        dipole_unit_vector = spherical_to_cartesian(
            (dipole_colatitude_rad, dipole_longitude_rad)
        )
        # shape (xyz=3, n_batches)
        dipole_vector = dipole_amplitude.reshape((1, n_batches)) * dipole_unit_vector
        poisson_mean = mean_count.reshape((n_batches,1)) * (
            1 + np.einsum('ji,jk', dipole_vector, pixel_vectors)
        )
        return poisson_mean

    def log_likelihood(self, theta: dict[str, NDArray]) -> NDArray:
        assert self.reference_data is not None
        assert self.reference_mask is not None

        dipole_signal = self.dipole_signal(**theta)
        assert dipole_signal.ndim == 2, (
            'dipole_signal must include a batch dimension.'
        )

        fine_mask = self.reference_mask.astype(np.bool_)
        if not np.any(fine_mask):
            raise ValueError(
                'reference_mask removes all pixels; cannot compute likelihood.'
            )

        native_npix = fine_mask.shape[0]
        assert dipole_signal.shape[1] == native_npix, (
            'Model output must match the native mask length.'
        )

        if self.downscale_nside:
            model_mask = np.broadcast_to(fine_mask, dipole_signal.shape).copy()
            dipole_signal, model_mask = downgrade_ignore_nan(
                dipole_signal,
                model_mask,
                self.downscale_nside
            )

            if model_mask.ndim == 2:
                first_mask = model_mask[0].astype(np.bool_)
                if not np.all(model_mask == first_mask):
                    raise ValueError(
                        'Mask differs across batches after downscaling.'
                    )
                effective_mask = first_mask
            else:
                effective_mask = model_mask.astype(np.bool_)

            coarse_npix = effective_mask.shape[0]
            assert self.reference_data.shape[0] == coarse_npix, (
                'reference_data must match the downscaled resolution.'
            )
            reference_data = self.reference_data
        else:
            effective_mask = fine_mask
            assert self.reference_data.shape[0] == native_npix, (
                'reference_data must match the native resolution when no '\
                'downscaling is configured.'
            )
            reference_data = self.reference_data

        if not np.any(effective_mask):
            raise ValueError(
                'Effective mask removes all pixels; cannot compute likelihood.'
            )

        observed_counts = reference_data[effective_mask]
        model_counts = dipole_signal[:, effective_mask]

        logpmf = sp.stats.poisson.logpmf(k=observed_counts, mu=model_counts)
        return np.sum(logpmf, axis=1)

# class SkyMap:
#     def __init__(self, nside: int = 32, device: str = 'cpu'):
#         self.nside = nside
#         self.device = device
#         self.mask = Mask(self.nside)
#         self.mask_map = torch.zeros(self.mask.npix)
#         self.nest = True
#
#     def configure(self,
#             dipole_method: Literal['base', 'poisson'],
#             dipole_hyperparameters: dict = {},
#             mask_kwargs: dict | None = None
#         ) -> None:
#         '''
#         Configures parameters for the batch simulator method.
#         '''
#         self.dipole_method = dipole_method
#         self.dipole_hyperparameters = dipole_hyperparameters
#         if mask_kwargs is None:
#             self.mask_kwargs = {'mask_fill_value': None}
#         else:
#             self.mask_kwargs = mask_kwargs
#
#     def generate_dipole(self, Theta: Tensor) -> None:
#         poisson_mean = self.dipole_signal(Theta)
#         self._density_map = poisson(poisson_mean)
#
#     def dipole_signal(self, Theta: Tensor) -> Tensor:
#         Theta = enforce_batchwise_input(Theta, ndim=4)
#
#         n_batches = Theta.shape[0]
#         pixel_indices = torch.arange(hp.nside2npix(self.nside))
#         pixel_vectors = torch.as_tensor(
#             torch.stack(
#                 hp.pix2vec(self.nside, pixel_indices, nest=True)
#             ),
#             device=self.device
#         ).to(torch.float32)
#         mean_count = Theta[:, 0]
#         dipole_amplitude = Theta[:, 1]
#         dipole_longitude = Theta[:, 2]
#         dipole_latitude = Theta[:, 3]
#         dipole_vector = dipole_amplitude.reshape((n_batches,1)) * spherical_to_cartesian(
#             (dipole_latitude, dipole_longitude),
#             device=self.device
#         )
#         poisson_mean = mean_count.reshape((n_batches,1)) * (
#             1 + torch.einsum('ij,jk', dipole_vector, pixel_vectors)
#         )
#         if n_batches == 1:
#             return poisson_mean.flatten()
#         else:
#             return poisson_mean
#
#     def generate_dipole_from_base(self,
#             observer_direction: tuple[float, float],
#             n_initial_points: int,
#             observer_speed: float = 0.00123,
#             luminosity_function_index: int = 2,
#             mean_spectral_index: float = 0.8,
#             sigma_spectral_index: float = 0.5,
#             flux_percentage_noise: float | str = 0.1,
#             minimum_flux_cut: float = 5,
#             noise_model_kwargs: dict = {}
#         ) -> Tensor:
#         '''
#         :param observer_direction: Direction of observer's motion in spherical
#             coordinates in radians (phi, theta).
#         '''
#         dipole_longitude_rad, dipole_colatitude_rad = torch.as_tensor(
#             observer_direction
#         )
#
#         dipole_longitude = torch.rad2deg(dipole_longitude_rad)
#         dipole_latitude = torch.rad2deg( torch.pi / 2 - dipole_colatitude_rad )
#
#         longitudes_deg, latitudes_deg,\
#         rest_fluxes, spectral_indices = sample_points_with_flux(
#             n_initial_points=n_initial_points,
#             luminosity_function_index=luminosity_function_index,
#             mean_spectral_index=mean_spectral_index,
#             sigma_spectral_index=sigma_spectral_index
#         )
#
#         self.boosted_longitudes_deg, self.boosted_latitudes_deg,\
#         boosted_fluxes = boost_points_with_flux(
#             longitudes_deg=longitudes_deg,
#             latitudes_deg=latitudes_deg,
#             rest_fluxes=rest_fluxes,
#             spectral_indices=spectral_indices,
#             observer_direction=(dipole_longitude, dipole_latitude),
#             observer_speed=observer_speed
#         )
#
#         noise_model = parse_noise_model(flux_percentage_noise)
#         noise_parameter = self.get_needed_noise_parameter(noise_model)
#         boosted_fluxes = add_noise_to_fluxes(
#             fluxes=boosted_fluxes,
#             noise_model=noise_model,
#             noise_scaling_parameter=noise_parameter,
#             **noise_model_kwargs
#         )
#
#         cut_boosted_longitudes_deg, cut_boosted_latitudes_deg,\
#         cut_boosted_fluxes = flux_cut(
#             minimum_flux=minimum_flux_cut,
#             longitudes=self.boosted_longitudes_deg,
#             latitudes=self.boosted_latitudes_deg,
#             fluxes=boosted_fluxes
#         )
#
#         self._density_map = self.make_density_map(
#             cut_boosted_longitudes_deg, cut_boosted_latitudes_deg
#         )
#         self.expected_amplitude = ellis_baldwin_amplitude(
#             observer_speed=observer_speed,
#             mean_spectral_index=mean_spectral_index,
#             luminosity_function_slope=luminosity_function_index
#         )
#         return self._density_map
#
#     def get_needed_noise_parameter(self, noise_model: Callable) -> Tensor:
#         if type(noise_model) is float:
#             return None
#         else:
#             model_to_parameter = {
#                 ecliptic_noise: self.get_ecliptic_latitudes()
#             }
#             return model_to_parameter[noise_model]
#
#     def get_ecliptic_latitudes(self) -> Tensor:
#         _, boosted_ecliptic_latitudes_deg = equatorial_to_ecliptic(
#             ra=self.boosted_longitudes_deg.numpy(),
#             dec=self.boosted_latitudes_deg.numpy(),
#             output_unit='degrees'
#         )
#         return torch.as_tensor(
#             boosted_ecliptic_latitudes_deg,
#             dtype=torch.float32
#         )
#
#     def make_density_map(self, longitudes: Tensor, latitudes: Tensor) -> None:
#         source_indices = hp.ang2pix(
#             self.nside, longitudes, latitudes, lonlat=True, nest=self.nest
#         )
#         return torch.bincount(
#             source_indices, minlength=hp.nside2npix(self.nside)
#         )
#
#     def mask_pixels(self, fill_value = None, **kwargs) -> None:
#         self.kwarg_to_mask = {'equator_mask': self.mask.equator_mask}
#         for key, val in kwargs.items():
#             masked_pixel_indices = self.kwarg_to_mask[key](val)
#             self.mask_map[masked_pixel_indices] = 1
#
#         if fill_value == None:
#             self.fill_value = torch.nan
#         else:
#             self.fill_value = fill_value
#
#     @property
#     def density_map(self):
#         out = self._density_map
#         out[self.mask_map == 1] = self.fill_value
#         return out
#
#     def batch_simulator(self,
#             proposal_distribution,
#             prior_returns_numpy: bool,
#             n_samples: int,
#             n_workers: int = 32
#     ) -> tuple[Tensor, Tensor]:
#         self.n_samples = n_samples
#
#         if self.dipole_method == 'poisson':
#             Theta, self.batch_density_maps = self.poisson_batches(
#                 proposal_distribution=proposal_distribution,
#                 n_samples=n_samples
#             )
#
#         elif self.dipole_method == 'base':
#
#             def simulator(Theta):
#                 N, v = Theta[0], Theta[1]
#                 phi, theta = Theta[2], Theta[3]
#                 int_N = N.to(torch.int32).item()
#
#                 density_map = self.generate_dipole_from_base(
#                     observer_direction=(phi, theta),
#                     n_initial_points=int_N,
#                     observer_speed=v,
#                     **self.dipole_hyperparameters
#                 )
#                 return density_map
#
#             simulator = process_simulator(
#                 simulator, proposal_distribution, prior_returns_numpy
#             )
#             check_sbi_inputs(simulator=simulator, prior=proposal_distribution)
#             Theta, self.batch_density_maps = simulate_for_sbi(
#                 simulator=simulator,
#                 proposal=proposal_distribution,
#                 num_workers=n_workers,
#                 num_simulations=n_samples
#             )
#         else:
#             raise Exception('Method not recognised.')
#
#         self.batchwise_mask(**self.mask_kwargs)
#
#         return Theta, self.batch_density_maps
#
#     def batchwise_mask(self, mask_fill_value, **mask_kwargs) -> None:
#         if mask_fill_value == None:
#             fill_value = torch.nan
#         else:
#             fill_value = mask_fill_value
#
#         self.mask_pixels(**mask_kwargs)
#         self.batch_mask_maps = self.mask_map.repeat((self.n_samples, 1))
#         self.batch_density_maps[self.batch_mask_maps == 1] = fill_value
#
#     def poisson_batches(self,
#             proposal_distribution,
#             n_samples: int
#     ) -> tuple[Tensor]:
#         Theta = proposal_distribution.sample((n_samples,))
#         poisson_mean = self.dipole_signal(Theta)
#         batch_density_maps = poisson(poisson_mean)
#         return (Theta, batch_density_maps)

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
