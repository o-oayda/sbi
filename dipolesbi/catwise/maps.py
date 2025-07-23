from astropy.table import Table
from dipolesbi.tools.utils import (
    Sample1DHistogram, ParameterMap, MultinomialSample2DHistogram
)
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
from scipy.stats import binned_statistic_2d
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import RegularGridInterpolator
from numpy.typing import NDArray
from datetime import datetime
import matplotlib.pyplot as plt

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

    def mask_pixels(self, fill_value = None, mask_north_ecliptic: bool = True) -> None:
        print('Masking pixels...')
        self.mask = Mask(nside=self.nside)
        self.mask_map = torch.zeros(self.mask.npix)
        masked_pixel_indices = set(self.mask.catwise_mask())

        if mask_north_ecliptic:
            north_pole_pixels = self.mask.north_ecliptic_mask()
            masked_pixel_indices.update(north_pole_pixels)
        
        self.masked_pixel_indices_set = masked_pixel_indices
        self.masked_pixel_indices_list = list(masked_pixel_indices)
        self.mask_map[self.masked_pixel_indices_list] = 1
        
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
        self.cat_w1_max = cat_w1_max
        self.cat_w12_min = cat_w12_min
        self.cut_path = self._get_cut_path(cat_w1_max, cat_w12_min)
        self.file_name = (
            f'catwise2020_corr_w12{self.cat_w12_min_str}_w1{self.cat_w1_max_str}.fits'
        )
        self.catalogue_is_loaded = False
    
    def _get_cut_path(self,
            cat_w1_max: float,
            cat_w12_min: float
        ) -> str:
        self.cat_w1_max_str = str(cat_w1_max).replace('.', 'p')
        self.cat_w12_min_str = str(cat_w12_min).replace('.', 'p')
        return f'{self.cat_w12_min_str}_{self.cat_w1_max_str}'

    def load_catalogue(self):
        self.file_path = f'dipolesbi/catwise/{self.file_name}'
        print('Loading CatWISE2020...')
        self.catalogue = Table.read(
            self.file_path,
            unit_parse_strict='silent' # supress unit warning printouts
        )
        print('Finished loading CatWISE2020.')
        self.catalogue_is_loaded = True
    
    def generate_dipole(self,
            n_initial_samples: int,
            w1_max: float = 16.4,
            w1_min: float = 9,
            w12_min: float = 0.8,
            observer_speed: float = CMB_BETA,
            dipole_longitude: float = CMB_L,
            dipole_latitude: float = CMB_B,
            error_scale: float = 1.0
        ) -> Tensor:
        self.observer_speed = observer_speed
        self.dipole_longitude = dipole_longitude
        self.dipole_latitude = dipole_latitude

        self.n_samples = n_initial_samples
        rest_w1_samples, rest_w2_samples = self.sample_magnitudes(self.n_samples)

        # since w12 sets alpha, it is wrong to draw alpha independently from 
        # the empirical distribuion; instead, use lookups
        rest_w12_samples = rest_w1_samples - rest_w2_samples
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

        # mask now for efficiency
        source_pixel_indices = hp.ang2pix(
            self.nside,
            boosted_source_longitudes_deg,
            boosted_source_latitudes_deg,
            lonlat=True,
            nest=True
        )
        self.mask_pixels()
        is_masked = self.mask_map == 1
        masked_pixel_indices = torch.nonzero(is_masked).squeeze()
        mask_slice = ~torch.isin(
            source_pixel_indices,
            masked_pixel_indices
        )
        rest_w1_samples = rest_w1_samples[mask_slice]
        rest_w2_samples = rest_w2_samples[mask_slice]
        rest_w12_samples = rest_w12_samples[mask_slice]
        boosted_source_longitudes_deg = boosted_source_longitudes_deg[mask_slice]
        boosted_source_latitudes_deg = boosted_source_latitudes_deg[mask_slice]
        rest_source_to_dipole_angle_deg = rest_source_to_dipole_angle_deg[mask_slice]
        spectral_indices = spectral_indices[mask_slice]
        source_pixel_indices = source_pixel_indices[mask_slice]

        self.boosted_w1_samples = self.boost_magnitudes(
            rest_w1_samples, rest_source_to_dipole_angle_deg, spectral_indices
        )
        self.boosted_w2_samples = self.boost_magnitudes(
            rest_w2_samples, rest_source_to_dipole_angle_deg, spectral_indices
        )
        
        self.source_log_w1_coverages = np.log10(self.w1_coverage_map[source_pixel_indices])
        self.source_log_w2_coverages = np.log10(self.w2_coverage_map[source_pixel_indices])
        self.w1_error = torch.as_tensor(
            self.w1mag_coverage_rgi(
                torch.column_stack(
                    [self.boosted_w1_samples, self.source_log_w1_coverages]
                )
            )
        )
        self.w2_error = torch.as_tensor(
            self.w2mag_coverage_rgi(
                torch.column_stack(
                    [self.boosted_w2_samples, self.source_log_w2_coverages]
                )
            )
        )
        self.boosted_w1_samples, self.boosted_w2_samples = self.add_error(
            w1_magnitudes=self.boosted_w1_samples,
            w2_magnitudes=self.boosted_w2_samples,
            w1_error=error_scale * self.w1_error,
            w2_error=error_scale * self.w2_error
        )

        # should be identical to rest_w12_samples since colour is invariant;
        # TODO: check this
        boosted_w12_samples = self.boosted_w1_samples - self.boosted_w2_samples
        boosted_w12_errors = np.sqrt( self.w1_error**2 + self.w2_error**2 )
        
        cut = self.magnitude_cut_boolean(
            w1_magnitudes=self.boosted_w1_samples,
            w12_magnitudes=boosted_w12_samples,
            w1_max=w1_max,
            w1_min=w1_min,
            w12_min=w12_min
        )
        cut_boosted_source_longitudes_deg = boosted_source_longitudes_deg[cut]
        cut_boosted_source_latitudes_deg = boosted_source_latitudes_deg[cut]
        cut_boosted_w1_samples = self.boosted_w1_samples[cut]
        cut_boosted_w2_samples = self.boosted_w2_samples[cut]
        cut_boosted_w12_samples = boosted_w12_samples[cut]
        cut_boosted_w12_errors = boosted_w12_errors[cut]
        cut_source_pixel_indices = source_pixel_indices[cut]

        self._density_map = self.make_density_map(
            longitudes=cut_boosted_source_longitudes_deg,
            latitudes=cut_boosted_source_latitudes_deg
        )

        self.final_w1_samples = cut_boosted_w1_samples
        self.final_w2_samples = cut_boosted_w2_samples
        self.final_w12_samples = cut_boosted_w12_samples
        self.final_w12_frac_errors = cut_boosted_w12_errors / self.final_w12_samples
        self.final_pixel_indices = cut_source_pixel_indices

        return self.density_map
    
    def simulator(self, Theta):
        N, eta, v = Theta[0], Theta[1], Theta[2]
        phi, theta = Theta[3], Theta[4]
        int_N = N.to(torch.int32).item()

        density_map = self.generate_dipole(
            n_initial_samples=int_N,
            dipole_longitude=np.rad2deg(phi),
            dipole_latitude=np.rad2deg(np.pi / 2 - theta),
            observer_speed=v,
            error_scale=eta
        )
        return density_map

    def batch_simulator(self,
            proposal_distribution,
            prior_returns_numpy: bool,
            n_samples: int,
            n_workers: int = 32
        ) -> tuple[Tensor, Tensor]:
        self.n_samples = n_samples

        simulator = self.simulator

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

    def make_density_map(self, longitudes: Tensor, latitudes: Tensor) -> Tensor:
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

    def mask_pixels(self,
            fill_value = None,
            mask_north_ecliptic: bool = True
    ) -> None:
        self.mask = Mask(nside=self.nside)
        self.mask_map = torch.zeros(self.mask.npix)

        assert self.nside == 64, 'CatWISE mask requires nside=64.'
        masked_pixel_indices = set(self.mask.catwise_mask())
        
        if mask_north_ecliptic:
            north_pole_pixels = self.mask.north_ecliptic_mask()
            masked_pixel_indices.update(north_pole_pixels)
        
        self.masked_pixel_indices_set = masked_pixel_indices
        self.masked_pixel_indices_list = list(masked_pixel_indices)
        self.mask_map[self.masked_pixel_indices_list] = 1

        if fill_value == None:
            self.fill_value = torch.nan
        else:
            self.fill_value = fill_value

    def add_error(self,
            w1_magnitudes: Tensor,
            w1_error: Tensor,
            w2_magnitudes: Tensor,
            w2_error: Tensor
        ) -> tuple[Tensor, Tensor]:
        boosted_w1_samples = torch.normal(
            mean=w1_magnitudes,
            std=w1_error
        )
        boosted_w2_samples = torch.normal(
            mean=w2_magnitudes,
            std=w2_error
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

    def precompute_data(self, no_check: bool = False):
        # load catalogue and mask
        if not self.catalogue_is_loaded:
            self.load_catalogue()
        self.mask_pixels()
        self.make_masked_catalogue()

        self.create_w1_w2_distribution()
        # self.create_spectral_index_distribution(no_check=no_check)
        # self.create_error_map()
        self.create_coverage_maps()
        self.create_magnitude_coverage_function()
    
    def make_masked_catalogue(self):
        assert self.catalogue_is_loaded, 'Load catalogue first.'
        assert hasattr(self, 'masked_pixel_indices_set'), 'Load mask first.'
        
        print('Generating masked catalogue...')
        all_pixel_indices = hp.ang2pix(
            self.nside,
            self.catalogue['l'],
            self.catalogue['b'],
            lonlat=True,
            nest=True
        )
        self.catalogue_mask = [
            idx not in self.masked_pixel_indices_set
            for idx in all_pixel_indices
        ]
        self.masked_catalogue = self.catalogue[self.catalogue_mask]
        print('Done.')

    def create_coverage_maps(self):
        pixel_indices = hp.ang2pix(
            self.nside,
            self.masked_catalogue['l'],
            self.masked_catalogue['b'],
            lonlat=True,
            nest=True
        )
        print('Building coverage maps...')
        w1_covmap = ParameterMap(
            pixel_indices=pixel_indices,
            parameter=self.masked_catalogue['w1cov'],
            nside=self.nside
        ).get_map()
        w2_covmap = ParameterMap(
            pixel_indices=pixel_indices,
            parameter=self.masked_catalogue['w2cov'],
            nside=self.nside
        ).get_map()
        
        file_path = f'dipolesbi/catwise/{self.cut_path}/data/coverage_map/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        torch.save(
            torch.as_tensor(w1_covmap),
            f'{file_path}w1_coverage_map.pt'
        )
        torch.save(
            torch.as_tensor(w2_covmap),
            f'{file_path}w2_coverage_map.pt'
        )
        print(f'Saved coverage maps at {file_path}.')

        plt.figure()
        hp.projview(w1_covmap, nest=True, norm='log')
        plt.savefig(f'{file_path}w1_coverage_map.png', dpi=300)

        plt.figure()
        hp.projview(w2_covmap, nest=True, norm='log')
        plt.savefig(f'{file_path}w2_coverage_map.png', dpi=300)

        print(f'Saved coverage map figures at {file_path}.')

    def create_magnitude_coverage_function(self):
        N_1D_BINS = 200

        # define magnitude-coverage grid bins, same for w1 and w2 for simplicity
        magnitude_bins = np.linspace(
            np.min(self.masked_catalogue['w1']),
            self.cat_w1_max,
            N_1D_BINS
        )
        coverage_bins = np.linspace(1.5, 4., N_1D_BINS)
        magnitude_centres = 0.5 * (magnitude_bins[:-1] + magnitude_bins[1:])
        coverage_centres = 0.5 * (coverage_bins[:-1] + coverage_bins[1:])

        for band in ['w1', 'w2']:
            print(f'Building {band} mag-coverage-error relation...')
                
            # compute median raw photometric across all sources in each cell
            median_error_grid, x_edges, y_edges, _ = binned_statistic_2d(
                self.masked_catalogue[f'{band}'],
                np.log10(self.masked_catalogue[f'{band}cov']),
                self.masked_catalogue[f'{band}e'],
                statistic='median',
                bins=[magnitude_bins, coverage_bins] # type: ignore
            )
            n_sources, *_ = binned_statistic_2d(
                self.masked_catalogue[f'{band}'],
                np.log10(self.masked_catalogue[f'{band}cov']),
                self.masked_catalogue[f'{band}e'],
                statistic='count',
                bins=[magnitude_bins, coverage_bins] # type: ignore
            )

            # to remove noisy cells
            median_error_grid[n_sources < 10] = np.nan

            # do nearest neighbour interpolation to fill nan cells
            magnitude_grid, coverage_grid = np.meshgrid(
                magnitude_centres, coverage_centres, indexing='ij'
            )
            mask = ~np.isnan(median_error_grid)
            valid_indices = np.where(mask)
            X_train = np.column_stack(
                [magnitude_grid[valid_indices], coverage_grid[valid_indices]]
            )
            y_train = median_error_grid[valid_indices]

            nbrs = NearestNeighbors(
                n_neighbors=4,
                algorithm='kd_tree',
                leaf_size=30
            )
            nbrs.fit(X_train)

            def knn_interpolate(X_pred, nbrs: NearestNeighbors):
                '''
                KNN interpolation with inverse distance weighting.
                '''
                distances, indices = nbrs.kneighbors(X_pred)
                
                # Inverse distance weighting
                weights = 1 / (distances + 1e-8) # Eps. to avoid division by zero
                weights = weights / weights.sum(axis=1)[:, np.newaxis]
                
                # Weighted prediction
                return np.sum(weights * y_train[indices], axis=1)
            
            filled_median_error_grid = median_error_grid.copy()
            nan_indices = np.where(~mask)
            X_predict = np.column_stack(
                [magnitude_grid[nan_indices], coverage_grid[nan_indices]]
            )
            filled_errors = knn_interpolate(X_predict, nbrs)
            filled_median_error_grid[~mask] = filled_errors

            rgi = RegularGridInterpolator(
                (magnitude_centres, coverage_centres), 
                filled_median_error_grid,
                method='linear',
                bounds_error=False,
                fill_value=None # extrapolation for 16.95 -> 17 mag # type: ignore
            )
            file_path = f'dipolesbi/catwise/{self.cut_path}/data/mag_coverage'
            self.save_interpolator(
                band=band,
                interpolator=rgi,
                mag_bins=magnitude_bins,
                cov_bins=coverage_bins,
                filled_grid=filled_median_error_grid,
                file_path=file_path
            )
            plt.figure()
            plt.pcolormesh(
                magnitude_bins,
                coverage_bins,
                filled_median_error_grid.T,
                shading='auto'
            )
            plt.colorbar()
            plt.savefig(f'{file_path}/{band}_matrix_plot.png', dpi=300)

    def save_interpolator(self,
            band: str,
            interpolator: RegularGridInterpolator,
            mag_bins: NDArray[np.float_],
            cov_bins: NDArray[np.float_],
            filled_grid: NDArray[np.float_],
            file_path: str
    ) -> bool:
        '''
        Save RegularGridInterpolator and metadata for use in batch simulations.
        '''
        save_data = {
            'interpolator': interpolator,
            'band': band,
            'mag_bins': mag_bins,
            'cov_bins': cov_bins,
            'filled_grid': filled_grid,
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'mag_range': (mag_bins.min(), mag_bins.max()),
                'cov_range': (cov_bins.min(), cov_bins.max()),
                'grid_shape': filled_grid.shape,
                'interpolation_method': 'linear'
            }
        }
        try:
            full_path = f'{file_path}/{band}_median_error_interpolator.pkl'

            if not os.path.exists(file_path):
                os.makedirs(file_path)

            with open(full_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Interpolator saved to: {full_path}")
            return True
        
        except Exception as e:
            print(f"✗ Error saving interpolator: {e}")
            return False
        
    def load_interpolator(self, full_path: str) -> RegularGridInterpolator:
        try:
            with open(full_path, 'rb') as f:
                save_data = pickle.load(f)

            print(f"Interpolator loaded from: {full_path}")
            print(f"Created: {save_data['metadata']['creation_date']}")
            print(f"Mag range: {save_data['metadata']['mag_range']}")
            print(f"Cov range: {save_data['metadata']['cov_range']}")
            print(f"Grid shape: {save_data['metadata']['grid_shape']}")
            
            return save_data['interpolator']
        
        except Exception as e:
            print(f"Error loading interpolator: {e}")
            raise Exception(e)

    def initialise_data(self):
        self.colour_mag_sampler = MultinomialSample2DHistogram()
        self.colour_mag_sampler.load_data(
            f'dipolesbi/catwise/{self.cut_path}/data/colour_mag/'
        )
        self.w1mag_coverage_rgi = self.load_interpolator(
            f'dipolesbi/catwise/{self.cut_path}/data/'
            'mag_coverage/w1_median_error_interpolator.pkl'
        )
        self.w2mag_coverage_rgi = self.load_interpolator(
            f'dipolesbi/catwise/{self.cut_path}/data/'
            'mag_coverage/w2_median_error_interpolator.pkl'
        )
        path = f'dipolesbi/catwise/{self.cut_path}/data/coverage_map/'
        self.w1_coverage_map = torch.load(
            f'{path}w1_coverage_map.pt'
        )
        self.w2_coverage_map = torch.load(
            f'{path}w2_coverage_map.pt'
        )

    def create_error_map(self) -> None:
        assert self.catalogue_is_loaded

        l, b = self.masked_catalogue['l'], self.masked_catalogue['b']
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
                self.masked_catalogue['w1e'][active_pixel]
                / self.masked_catalogue['w1'][active_pixel]
            )
            w2_fractional_error = (
                self.masked_catalogue['w2e'][active_pixel]
                / self.masked_catalogue['w2'][active_pixel]
            )
            w12_fractional_error = (
                self.masked_catalogue['w12e'][active_pixel]
                / self.masked_catalogue['w12'][active_pixel]
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

    def create_w1_w2_distribution(self,
            bins: int = 200,
            **hist_kwargs
        ) -> None:
        assert self.catalogue_is_loaded
        
        w1_mags = self.masked_catalogue['w1']
        w2_mags = self.masked_catalogue['w2']

        sampler = MultinomialSample2DHistogram()
        sampler.build(
            w1_mags,
            w2_mags,
            **{
                'bins': bins,
                **hist_kwargs
            }
        )
        path = f'dipolesbi/catwise/{self.cut_path}/data/colour_mag/'
        sampler.save_data(path)
        print(f'Saved W1-W2 distribution to {path}.')

        print(f'Generating W1-W2 distribution plot...')
        w1_samples, w2_samples = sampler.sample(n_samples=30_000_000)
        plt.hist2d(w1_samples, w2_samples, bins=400, norm='log')
        plt.savefig(f'{path}w1_w2_dist.png', dpi=300)
    
    def create_spectral_index_distribution(self,
            bins: int = 200,
            no_check: bool = False,
            **hist_kwargs
        ) -> None:
        assert self.catalogue_is_loaded

        lookup = AlphaLookup()
        out_table = lookup.make_alpha(
            self.masked_catalogue['w1'], self.masked_catalogue['w12'], no_check=no_check
        )
        self.spectral_indices = out_table['alpha_W1'].data # type: ignore
        
        sampler = Sample1DHistogram()
        sampler.build(self.spectral_indices,
            **{
                'bins': bins,
                **hist_kwargs
            }
        )
        sampler.save_data(f'dipolesbi/catwise/{self.cut_path}/data/spectral_index/')
    
    def resample_catwise_magnitudes(self, n_samples: int) -> tuple[Tensor, Tensor]:
        if not self.catalogue_is_loaded:
            self.load_catalogue()
        
        if not hasattr(self, 'masked_catalogue'):
            self.mask_pixels()
            self.make_masked_catalogue()
        
        w1_real, w2_real = self.masked_catalogue['w1'], self.masked_catalogue['w2']
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
        ) -> tuple[Tensor, Tensor, Tensor]:
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