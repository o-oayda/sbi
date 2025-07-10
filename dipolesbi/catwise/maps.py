from astropy.table import Table
from dipolesbi.tools.utils import Sample1DHistogram, Sample2DHistogram
from torch.types import Tensor
import torch
from dipolesbi.tools.physics import (
    sample_spherical_points, aberrate_points, boost_magnitudes
)
from dipolesbi.tools.constants import *
from dipolesbi.catwise.utils import AlphaLookup
import numpy as np
import healpy as hp
from tqdm import tqdm
import os
from dipolesbi.tools.maps import Mask
from sbi.utils.user_input_checks import process_simulator, check_sbi_inputs
from sbi.inference import simulate_for_sbi
from collections import defaultdict
import pickle

class CatwiseReal:
    # TODO: refactor due to code duplication with CatwiseSim
    def __init__(self, nside: int = 64):
        self.file_path = (
            'dipolesbi/catwise/catwise_agns_masked_final_w1lt16p5_alpha.fits'
        )
        self.nside = nside

        self.load_catalogue()
        self.make_cuts()
        self.make_density_map()
        self.mask_pixels()
    
    def load_catalogue(self) -> None:
        print(f'Reading in CatWISE2020 from {self.file_path}...')
        self.catalogue = Table.read(self.file_path)
        print(f'Loaded CatWISE2020.')


    def make_cuts(self) -> None:
        print('Making flux cuts...')
        flux_cuts = (self.catalogue['w1'] < 16.4) & (self.catalogue['w1'] > 9)
        self.catalogue = self.catalogue[flux_cuts]

    def make_density_map(self) -> None:
        print('Computing density map...')
        source_indices = hp.ang2pix(
            self.nside,
            self.catalogue['l'].data,
            self.catalogue['b'].data,
            lonlat=True,
            nest=True
        )
        self.source_indices = torch.as_tensor(source_indices)
        self._density_map =  torch.bincount(
            self.source_indices,
            minlength=hp.nside2npix(self.nside)
        )
    
    @property
    def density_map(self):
        out = self._density_map.to(dtype=torch.float32)
        out[self.mask_map == 1] = self.fill_value
        return out

    def mask_pixels(self, fill_value = None) -> None:
        print('Masking pixels...')
        self.mask = Mask(nside=self.nside)
        self.mask_map = torch.zeros(self.mask.npix)

        masked_pixel_indices = self.mask.catwise_mask()
        self.mask_map[masked_pixel_indices] = 1
        
        if fill_value == None:
            self.fill_value = torch.nan
        else:
            self.fill_value = fill_value

class CatwiseSim:
    def __init__(self,
            cat_w1_max: float,
            cat_w12_min: float,
            nside: int = 64,
        ):
        self.nside = nside
        self.dipole_longitude = CMB_L
        self.dipole_latitude = CMB_B
        self.observer_speed = CMB_BETA
        self.cut_path = self._get_cut_path(cat_w1_max, cat_w12_min)
        self.file_name = (
            f'catwise2020_corr_w12-{self.cat_w12_min}_w1-{self.cat_w1_max}.fits'
        )
        # self.n_samples = 35_947_376 # length of above table
        self.catalogue_is_loaded = False
    
    def _get_cut_path(self,
            cat_w1_max: float,
            cat_w12_min: float
        ) -> str:
        self.cat_w1_max = str(cat_w1_max).replace('.', 'p')
        self.cat_w12_min = str(cat_w12_min).replace('.', 'p')
        return f'{self.cat_w12_min}_{self.cat_w1_max}'

    def load_catalogue(self):
        self.file_path = f'dipolesbi/catwise/{self.file_name}'
        print('Loading CatWISE2020...')
        self.catalogue = Table.read(self.file_path)
        print('Finished loading CatWISE2020.')
        self.catalogue_is_loaded = True
    
    def generate_dipole(self,
            n_initial_samples: int,
            w1_max: float = 16.4,
            w1_min: float = 9,
            w12_min: float = 0.8,
            observer_speed: float = CMB_BETA,
            dipole_longitude: float = CMB_L,
            dipole_latitude: float = CMB_B
        ) -> Tensor:
        self.observer_speed = observer_speed
        self.dipole_longitude = dipole_longitude
        self.dipole_latitude = dipole_latitude

        self.n_samples = n_initial_samples
        # rest_w1_samples, rest_w2_samples = self.sample_magnitudes(self.n_samples)
        # rest_w1_samples, rest_w2_samples = self.resample_catwise_magnitudes(self.n_samples)
        rest_w1_samples, rest_w12_samples = self.resample_colour_mag_distribution(
            n_samples=self.n_samples
        )

        # since w12 sets alpha, it is wrong to draw alpha independently from 
        # the empirical distribuion; instead, use lookups
        # spectral_indices = torch.as_tensor(
            # -self.spectral_index_sampler.sample(self.n_samples)
        # )
        # rest_w12_samples = rest_w1_samples - rest_w2_samples
        lookup = AlphaLookup()
        out_table = lookup.make_alpha(
            w1_magnitude=rest_w1_samples.numpy(),
            w12_color=rest_w12_samples.numpy(),
            no_check=True
        )
        spectral_indices = torch.as_tensor(
            out_table['alpha_W1'].data.astype('float32') # type: ignore
        )

        rest_source_longitudes_deg,\
        rest_source_latitudes_deg = self.sample_points(self.n_samples)
        
        boosted_source_longitudes_deg, boosted_source_latitudes_deg,\
        rest_source_to_dipole_angle_deg = self.aberrate_points(
            rest_source_longitudes_deg, rest_source_latitudes_deg
        )

        boosted_w1_samples = self.boost_magnitudes(
            rest_w1_samples, rest_source_to_dipole_angle_deg, spectral_indices
        )
        # dummy w2 samples
        boosted_w2_samples = torch.ones(len(boosted_w1_samples))
        # boosted_w2_samples = self.boost_magnitudes(
            # rest_w2_samples, rest_source_to_dipole_angle_deg, spectral_indices
        # )

        source_pixel_indices = hp.ang2pix(
            self.nside,
            boosted_source_longitudes_deg,
            boosted_source_latitudes_deg,
            lonlat=True,
            nest=True
        )
        # self.w1_fractional_error = self.w1_error_map[source_pixel_indices]
        # self.w2_fractional_error = self.w2_error_map[source_pixel_indices]
        self.w1_fractional_error,\
        self.w2_fractional_error = self.sample_errors_ultra_fast(
            source_pixel_indices
        )
        boosted_w1_samples, boosted_w2_samples,\
            sigma_w1, sigma_w2 = self.add_error(
            w1_magnitudes=boosted_w1_samples,
            w2_magnitudes=boosted_w2_samples,
            w1_fractional_error=self.w1_fractional_error,
            w2_fractional_error=self.w2_fractional_error
        )

        # boosted_w12_samples = boosted_w1_samples - boosted_w2_samples
        # boosted_w12_errors = np.sqrt( sigma_w1**2 + sigma_w2**2 )
        boosted_w12_samples = rest_w12_samples
        
        cut = self.magnitude_cut_boolean(
            w1_magnitudes=boosted_w1_samples,
            w12_magnitudes=boosted_w12_samples,
            w1_max=w1_max,
            w1_min=w1_min,
            w12_min=w12_min
        )
        cut_boosted_source_longitudes_deg = boosted_source_longitudes_deg[cut]
        cut_boosted_source_latitudes_deg = boosted_source_latitudes_deg[cut]
        cut_boosted_w1_samples = boosted_w1_samples[cut]
        cut_boosted_w2_samples = boosted_w2_samples[cut]
        cut_boosted_w12_samples = boosted_w12_samples[cut]
        cut_boosted_w12_errors = boosted_w12_errors[cut]

        self._density_map = self.make_density_map(
            longitudes=cut_boosted_source_longitudes_deg,
            latitudes=cut_boosted_source_latitudes_deg
        )
        self.mask_pixels()

        cut_source_pixel_indices = hp.ang2pix(
            self.nside,
            cut_boosted_source_longitudes_deg,
            cut_boosted_source_latitudes_deg,
            lonlat=True,
            nest=True
        )
        cut_masked_pixels = torch.isin(
            cut_source_pixel_indices,
            torch.nonzero(self.mask_map == 1).squeeze()
        )
        self.final_w1_samples = cut_boosted_w1_samples[~cut_masked_pixels]
        self.final_w2_samples = cut_boosted_w2_samples[~cut_masked_pixels]
        self.final_w12_samples = cut_boosted_w12_samples[~cut_masked_pixels]
        self.final_w12_frac_errors = (
            cut_boosted_w12_errors[~cut_masked_pixels]
            / self.final_w12_samples
        )
        self.final_pixel_indices = cut_source_pixel_indices[~cut_masked_pixels]

        return self.density_map

    def batch_simulator(self,
            proposal_distribution,
            prior_returns_numpy: bool,
            n_samples: int,
            n_workers: int = 32
        ) -> tuple[Tensor, Tensor]:
        self.n_samples = n_samples

        def simulator(Theta):
            N, v = Theta[0], Theta[1]
            phi, theta = Theta[2], Theta[3]
            int_N = N.to(torch.int32).item()

            density_map = self.generate_dipole(
                n_initial_samples=int_N,
                dipole_longitude=np.rad2deg(phi),
                dipole_latitude=np.rad2deg(np.pi / 2 - theta),
                observer_speed=v
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
        
        return Theta, self.batch_density_maps

    def make_density_map(self, longitudes: Tensor, latitudes: Tensor) -> None:
        source_indices = hp.ang2pix(
            self.nside, longitudes, latitudes, lonlat=True, nest=True
        )
        return torch.bincount(
            source_indices, minlength=hp.nside2npix(self.nside)
        )
    
    @property
    def density_map(self):
        out = self._density_map.to(dtype=torch.float32)
        out[self.mask_map == 1] = self.fill_value
        return out

    def mask_pixels(self, fill_value = None, **kwargs) -> None:
        self.mask = Mask(nside=self.nside)
        self.mask_map = torch.zeros(self.mask.npix)

        masked_pixel_indices = self.mask.catwise_mask()
        self.mask_map[masked_pixel_indices] = 1

        if fill_value == None:
            self.fill_value = torch.nan
        else:
            self.fill_value = fill_value

    def add_error(self,
            w1_magnitudes: Tensor,
            w1_fractional_error: Tensor,
            w2_magnitudes: Tensor,
            w2_fractional_error: Tensor
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sigma_w1 = w1_fractional_error * w1_magnitudes
        sigma_w2 = w2_fractional_error * w2_magnitudes
        
        boosted_w1_samples = torch.normal(
            mean=w1_magnitudes,
            std=sigma_w1
        )
        boosted_w2_samples = torch.normal(
            mean=w2_magnitudes,
            std=sigma_w2
        )
        return boosted_w1_samples, boosted_w2_samples, sigma_w1, sigma_w2
    
    def magnitude_cut_boolean(self,
            w1_magnitudes: Tensor,
            w12_magnitudes: Tensor,
            w1_max: float,
            w1_min: float,
            w12_min: float
        ) -> Tensor:
        condition = torch.logical_and(
            torch.logical_and(
                w1_magnitudes < w1_max,
                w1_magnitudes > w1_min
            ),
            w12_magnitudes > w12_min
        )
        # condition = (
            #   w1_magnitudes < w1_max
            # & w1_magnitudes > w1_min
            # & w12_magnitudes > w12_min
        # )
        return condition

    def precompute_data(self, no_check: bool = False):
        self.create_colour_magnitude_distribution()
        self.create_spectral_index_distribution(no_check=no_check)
        self.create_error_map()

    def initialise_data(self):
        self.colour_mag_sampler = Sample2DHistogram()
        self.colour_mag_sampler.load_data(
            f'dipolesbi/catwise/{self.cut_path}/data/colour_mag/'
        )
        self.spectral_index_sampler = Sample1DHistogram()
        self.spectral_index_sampler.load_data(
            f'dipolesbi/catwise/{self.cut_path}/data/spectral_index/'
        )

        path = f'dipolesbi/catwise/{self.cut_path}/data/error_map/'
        self.w1_error_map = torch.load(
            f'{path}w1_error_map.pt'
        )
        self.w2_error_map = torch.load(
            f'{path}w2_error_map.pt'
        )
        
        with open(f'{path}w1_error_dict.pt', 'rb') as f:
            self.w1_error_dict = pickle.load(f)
        
        with open(f'{path}w2_error_dict.pt', 'rb') as f:
            self.w2_error_dict = pickle.load(f)
        
        print('Creating fractional error lookups...')
        self.create_error_lookup_arrays()
        print('Done.')

    def create_error_map(self) -> None:
        assert self.catalogue_is_loaded

        l, b = self.catalogue['l'], self.catalogue['b']
        source_pixel_indices = hp.ang2pix(self.nside, l, b, lonlat=True, nest=True)
        n_pixels = hp.nside2npix(self.nside)
        
        w1_error_map = np.empty(n_pixels)
        w2_error_map = np.empty(n_pixels)
        w12_error_map = np.empty(n_pixels)
        w1_error_dict = defaultdict(list)
        w2_error_dict = defaultdict(list)
        w12_error_dict = defaultdict(list)

        for pix_ind in tqdm(range(n_pixels)):
            active_pixel = source_pixel_indices == pix_ind

            w1_fractional_error = (
                self.catalogue['w1e'][active_pixel] / self.catalogue['w1'][active_pixel]
            )
            w2_fractional_error = (
                self.catalogue['w2e'][active_pixel] / self.catalogue['w2'][active_pixel]
            )
            w12_fractional_error = (
                self.catalogue['w12e'][active_pixel] / self.catalogue['w12'][active_pixel]
            )

            w1_error_dict[pix_ind] = w1_fractional_error
            w2_error_dict[pix_ind] = w2_fractional_error
            w12_error_dict[pix_ind] = w12_fractional_error
            w1_error_map[pix_ind] = np.median( w1_fractional_error )
            w2_error_map[pix_ind] = np.median( w2_fractional_error )
            w12_error_map[pix_ind] = np.median( w12_fractional_error )
        
        file_path = f'dipolesbi/catwise/{self.cut_path}/data/error_map/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        torch.save(
            torch.as_tensor(w1_error_map),
            f'{file_path}w1_error_map.pt'
        )
        torch.save(
            torch.as_tensor(w2_error_map),
            f'{file_path}w2_error_map.pt'
        )
        torch.save(
            torch.as_tensor(w12_error_map),
            f'{file_path}w12_error_map.pt'
        )
        with open(f'{file_path}w1_error_dict.pt', 'wb') as handle:
            pickle.dump(w1_error_dict, handle)
        
        with open(f'{file_path}w2_error_dict.pt', 'wb') as handle:
            pickle.dump(w2_error_dict, handle)

        with open(f'{file_path}w12_error_dict.pt', 'wb') as handle:
            pickle.dump(w12_error_dict, handle)

    def create_colour_magnitude_distribution(self,
            bins: int = 200,
            **hist_kwargs
        ) -> None:
        assert self.catalogue_is_loaded
        
        w1_mags = self.catalogue['w1']
        w2_mags = self.catalogue['w2']

        sampler = Sample2DHistogram()
        sampler.build(
            w1_mags,
            w2_mags,
            **{
                'bins': bins,
                **hist_kwargs
            }
        )
        sampler.save_data(f'dipolesbi/catwise/{self.cut_path}/data/colour_mag/')
    
    def create_spectral_index_distribution(self,
            bins: int = 200,
            no_check: bool = False,
            **hist_kwargs
        ) -> None:
        assert self.catalogue_is_loaded

        lookup = AlphaLookup()
        out_table = lookup.make_alpha(
            self.catalogue['w1'], self.catalogue['w12'], no_check=no_check
        )
        self.spectral_indices = out_table['alpha_W1'].data
        
        sampler = Sample1DHistogram()
        sampler.build(self.spectral_indices,
            **{
                'bins': bins,
                **hist_kwargs
            }
        )
        # sampler.save_data(f'dipolesbi/catwise/{self.cut_path}/data/spectral_index/')
    
    def resample_catwise_magnitudes(self, n_samples: int) -> tuple[Tensor, Tensor]:
        if not self.catalogue_is_loaded:
            self.load_catalogue()
        
        w1_real, w2_real = self.catalogue['w1'], self.catalogue['w2']
        resampled_indexes = np.random.choice(len(w1_real), n_samples)
        w1_resampled = w1_real[resampled_indexes].astype('float32')
        w2_resampled = w2_real[resampled_indexes].astype('float32')
        
        return torch.as_tensor(w1_resampled), torch.as_tensor(w2_resampled)
    
    def resample_colour_mag_distribution(self, n_samples: int) -> tuple[Tensor, Tensor]:
        if not self.catalogue_is_loaded:
            self.load_catalogue()
        
        w1_real, w12_real = self.catalogue['w1'], self.catalogue['w12']
        resampled_indexes = np.random.choice(len(w1_real), n_samples)
        w1_resampled = w1_real[resampled_indexes].astype('float32')
        w12_resampled = w12_real[resampled_indexes].astype('float32')

        return torch.as_tensor(w1_resampled), torch.as_tensor(w12_resampled)

    def sample_magnitudes(self, n_samples: int) -> tuple[Tensor, Tensor]:
        w1_samples, w2_samples = self.colour_mag_sampler.sample(n_samples)
        return torch.as_tensor(w1_samples), torch.as_tensor(w2_samples)
    
    def sample_points(self, n_points: int) -> tuple[Tensor, Tensor]:
        longitudes_deg, latitudes_deg = sample_spherical_points(n_points)
        return longitudes_deg, latitudes_deg
    
    def aberrate_points(self,
            longitudes_deg: Tensor,
            latitudes_deg: Tensor
        ) -> Tensor:
        boosted_longitudes_deg, boosted_latitudes_deg,\
        rest_source_to_dipole_angle = aberrate_points(
            rest_longitudes=longitudes_deg,
            rest_latitudes=latitudes_deg,
            observer_direction=(self.dipole_longitude, self.dipole_latitude),
            observer_speed=self.observer_speed
        )
        return (
            boosted_longitudes_deg, boosted_latitudes_deg,
            rest_source_to_dipole_angle
        )

    def boost_magnitudes(self,
            magnitudes: Tensor,
            rest_source_to_dipole_angle: Tensor,
            spectral_index: Tensor
        ) -> Tensor:
        boosted_magnitudes = boost_magnitudes(
            magnitudes=magnitudes,
            angle_to_source=rest_source_to_dipole_angle,
            observer_speed=self.observer_speed,
            spectral_index=spectral_index
        )
        return boosted_magnitudes

    def create_error_lookup_arrays(self):
        """Create lookup arrays for ultra-fast error sampling."""
        n_pixels = hp.nside2npix(self.nside)
        
        # Find maximum number of errors per pixel
        max_errors_w1 = max(len(errors) for errors in self.w1_error_dict.values())
        max_errors_w2 = max(len(errors) for errors in self.w2_error_dict.values())
        
        # Create padded arrays (pad with NaN for empty slots)
        self.w1_error_array = np.full((n_pixels, max_errors_w1), np.nan, dtype=np.float32)
        self.w2_error_array = np.full((n_pixels, max_errors_w2), np.nan, dtype=np.float32)
        self.w1_n_errors = np.zeros(n_pixels, dtype=np.int32)
        self.w2_n_errors = np.zeros(n_pixels, dtype=np.int32)
        
        # Fill arrays
        for pixel_idx, errors in self.w1_error_dict.items():
            n_errors = len(errors)
            if n_errors > 0:
                self.w1_error_array[pixel_idx, :n_errors] = errors
                self.w1_n_errors[pixel_idx] = n_errors
        
        for pixel_idx, errors in self.w2_error_dict.items():
            n_errors = len(errors)
            if n_errors > 0:
                self.w2_error_array[pixel_idx, :n_errors] = errors
                self.w2_n_errors[pixel_idx] = n_errors

    def sample_errors_ultra_fast(self, source_pixel_indices):
        """Ultra-fast error sampling using pre-computed arrays."""
        n_sources = len(source_pixel_indices)
        
        # Generate random indices for each source
        w1_random_indices = np.random.randint(0, self.w1_n_errors[source_pixel_indices])
        w2_random_indices = np.random.randint(0, self.w2_n_errors[source_pixel_indices])
        
        # Handle pixels with no errors (set random index to 0, will select NaN)
        w1_random_indices[self.w1_n_errors[source_pixel_indices] == 0] = 0
        w2_random_indices[self.w2_n_errors[source_pixel_indices] == 0] = 0
        
        # Advanced indexing to get errors
        w1_errors = self.w1_error_array[source_pixel_indices, w1_random_indices]
        w2_errors = self.w2_error_array[source_pixel_indices, w2_random_indices]
        
        # Replace NaN with 0.0
        w1_errors = np.nan_to_num(w1_errors, nan=0.0)
        w2_errors = np.nan_to_num(w2_errors, nan=0.0)
        
        return torch.as_tensor(w1_errors), torch.as_tensor(w2_errors)