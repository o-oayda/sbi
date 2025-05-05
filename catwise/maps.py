from astropy.table import Table
from tools.utils import Sample1DHistogram, Sample2DHistogram
from torch.types import Tensor
import torch
from tools.physics import (
    sample_spherical_points, aberrate_points, boost_magnitudes
)
from tools.constants import *
from catwise.utils import AlphaLookup
import numpy as np
import healpy as hp
from tqdm import tqdm
import os
from tools.maps import Mask

class CatwiseSim:
    def __init__(self, nside: int = 64):
        self.nside = nside
        self.dipole_longitude = CMB_L
        self.dipole_latitude = CMB_B
        self.observer_speed = CMB_BETA
        self.file_name = 'catwise2020_corr_w12-0p5_w1-17p0.fits'
        self.n_samples = 35_947_376 # length of above table
        self.catalogue_is_loaded = False
        
    def load_catalogue(self):
        self.file_path = f'catwise/{self.file_name}'
        print('Loading CatWISE2020...')
        self.catalogue = Table.read(self.file_path)
        print('Finished loading CatWISE2020.')
        self.catalogue_is_loaded = True
    
    def generate_dipole(self,
            w1_max: float = 16.4,
            w1_min: float = 9,
            w12_min: float = 0.8
        ) -> Tensor:
        rest_w1_samples, rest_w2_samples = self.sample_magnitudes(self.n_samples)
        spectral_indices = torch.as_tensor(
            -self.spectral_index_sampler.sample(self.n_samples)
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
        boosted_w2_samples = self.boost_magnitudes(
            rest_w2_samples, rest_source_to_dipole_angle_deg, spectral_indices
        )

        source_pixel_indices = hp.ang2pix(
            self.nside,
            boosted_source_longitudes_deg,
            boosted_source_latitudes_deg,
            lonlat=True,
            nest=True
        )
        w1_fractional_error = self.w1_error_map[source_pixel_indices]
        w2_fractional_error = self.w2_error_map[source_pixel_indices]
        boosted_w1_samples, boosted_w2_samples = self.add_error(
            w1_magnitudes=boosted_w1_samples,
            w2_magnitudes=boosted_w2_samples,
            w1_fractional_error=w1_fractional_error,
            w2_fractional_error=w2_fractional_error
        )

        boosted_w12_samples = boosted_w1_samples - boosted_w2_samples
        
        cut = self.magnitude_cut_boolean(
            w1_magnitudes=boosted_w1_samples,
            w12_magnitudes=boosted_w12_samples,
            w1_max=w1_max,
            w1_min=w1_min,
            w12_min=w12_min
        )
        cut_boosted_source_longitudes_deg = boosted_source_longitudes_deg[cut]
        cut_boosted_source_latitudes_deg = boosted_source_latitudes_deg[cut]

        self._density_map = self.make_density_map(
            longitudes=cut_boosted_source_longitudes_deg,
            latitudes=cut_boosted_source_latitudes_deg
        )
        self.mask_pixels()

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
        ) -> tuple[Tensor]:
        boosted_w1_samples = torch.normal(
            mean=w1_magnitudes,
            std=w1_fractional_error * w1_magnitudes
        )
        boosted_w2_samples = torch.normal(
            mean=w2_magnitudes,
            std=w2_fractional_error * w2_magnitudes
        )
        return boosted_w1_samples, boosted_w2_samples
    
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

    def precompute_data(self):
        self.create_colour_magnitude_distribution()
        self.create_spectral_index_distribution()
        self.create_error_map()

    def initialise_data(self):
        self.colour_mag_sampler = Sample2DHistogram()
        self.colour_mag_sampler.load_data('catwise/data/colour_mag/')
        self.spectral_index_sampler = Sample1DHistogram()
        self.spectral_index_sampler.load_data('catwise/data/spectral_index/')
        self.w1_error_map = torch.load('catwise/data/error_map/w1_error_map.pt')
        self.w2_error_map = torch.load('catwise/data/error_map/w2_error_map.pt')

    def create_error_map(self) -> None:
        assert self.catalogue_is_loaded

        l, b = self.catalogue['l'], self.catalogue['b']
        source_pixel_indices = hp.ang2pix(self.nside, l, b, lonlat=True, nest=True)
        n_pixels = hp.nside2npix(self.nside)
        
        w1_error_map = np.empty(n_pixels)
        w2_error_map = np.empty(n_pixels)

        for pix_ind in tqdm(range(n_pixels)):
            active_pixel = source_pixel_indices == pix_ind
            
            w1_fractional_error = np.median(
                self.catalogue['w1e'][active_pixel] / self.catalogue['w1'][active_pixel]
            )
            w2_fractional_error = np.median(
                self.catalogue['w2e'][active_pixel] / self.catalogue['w2'][active_pixel]
            )

            w1_error_map[pix_ind] = w1_fractional_error
            w2_error_map[pix_ind] = w2_fractional_error
        
        if not os.path.exists('catwise/data/error_map/'):
            os.makedirs('catwise/data/error_map/')
        
        torch.save(
            torch.as_tensor(w1_error_map),
            'catwise/data/error_map/w1_error_map.pt'
        )
        torch.save(
            torch.as_tensor(w2_error_map),
            'catwise/data/error_map/w2_error_map.pt'
        )

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
        sampler.save_data('catwise/data/colour_mag/')
    
    def create_spectral_index_distribution(self,
            bins: int = 200,
            **hist_kwargs
        ) -> None:
        assert self.catalogue_is_loaded

        lookup = AlphaLookup()
        out_table = lookup.make_alpha(self.catalogue['w1'], self.catalogue['w12'])
        self.spectral_indices = out_table['alpha_W1'].data
        
        sampler = Sample1DHistogram()
        sampler.build(self.spectral_indices,
            **{
                'bins': bins,
                **hist_kwargs
            }
        )
        sampler.save_data('catwise/data/spectral_index/')
    
    def sample_magnitudes(self, n_samples: int) -> tuple[Tensor]:
        w1_samples, w2_samples = self.colour_mag_sampler.sample(n_samples)
        return torch.as_tensor(w1_samples), torch.as_tensor(w2_samples)
    
    def sample_points(self, n_points: int) -> Tensor:
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
        ) -> None:
        boosted_magnitudes = boost_magnitudes(
            magnitudes=magnitudes,
            angle_to_source=rest_source_to_dipole_angle,
            observer_speed=self.observer_speed,
            spectral_index=spectral_index
        )
        return boosted_magnitudes