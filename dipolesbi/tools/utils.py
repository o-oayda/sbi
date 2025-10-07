from typing import Callable, Optional, cast
import healpy as hp
from joblib import Parallel, delayed
import numpy as np
import torch
from torch import poisson
from torch.types import Tensor
from scipy.interpolate import interp1d
import os
import sys
from astropy.coordinates import SkyCoord
import astropy.units as u
from operator import itemgetter
import dill as pickle
from numpy.typing import DTypeLike, NDArray
from collections import defaultdict
import torch.nn.functional as F
from jax import numpy as jnp
import jax
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.ui import MultiRoundInfererUI
from dipolesbi.tools.dataloader import healpix_map_dataset_idx, healpix_map_dataset
from tqdm import tqdm


def batch_simulate(
        theta: dict[str, NDArray],
        model_callable: Callable[..., tuple[NDArray, NDArray]],
        n_workers: int,
        ui: Optional[MultiRoundInfererUI] = None,
        rng_key: Optional[NPKey] = None
) -> tuple[NDArray, NDArray]:
    theta_np = {key: np.asarray(val) for key, val in theta.items()}

    def _leading_dim(arr: np.ndarray) -> Optional[int]:
        return None if arr.shape == () else arr.shape[0]

    n_simulations = 1
    for arr in theta_np.values():
        leading = _leading_dim(arr)
        if leading is not None:
            n_simulations = leading
            break

    def _slice_param(arr: np.ndarray, idx: int):
        if arr.shape == ():
            return arr
        return arr[idx]

    params_per_sim = [
        {key: _slice_param(arr, idx) for key, arr in theta_np.items()}
        for idx in range(n_simulations)
    ] if n_simulations > 1 else [
        {key: (arr if arr.shape == () else arr) for key, arr in theta_np.items()}
    ]

    sim_keys: list[Optional[NPKey]]
    if rng_key is not None:
        # Fold the base key with the simulation index so each task gets a unique,
        # order-independent seed even when joblib reorders execution.
        sim_keys = [rng_key.fold_in(idx) for idx in range(n_simulations)]
    else:
        sim_keys = [None] * n_simulations

    if n_simulations == 1:
        kwargs = params_per_sim[0]
        key = sim_keys[0]
        if key is not None:
            kwargs = {**kwargs, 'rng_key': key}
        return model_callable(**kwargs)

    def _run_single(idx: int, key: Optional[NPKey], kwargs: dict[str, NDArray]):
        call_kwargs = dict(kwargs)
        if key is not None:
            call_kwargs['rng_key'] = key
        return idx, model_callable(**call_kwargs)

    iterator = Parallel(return_as='generator', n_jobs=n_workers)(
        delayed(_run_single)(idx, key, kwargs)
        for idx, (key, kwargs) in enumerate(zip(sim_keys, params_per_sim))
    )

    progress = None
    if ui is not None:
        pass
        # if ui._global_prog is not None:
        #     ui.begin_global_progress(total=n_simulations)
    else:
        progress = tqdm(total=n_simulations)

    simulation_outputs: list[tuple[int, tuple[NDArray, NDArray]]] = []
    for idx, result in enumerate(iterator, start=1):
        simulation_outputs.append(result)
        if ui is not None:
            ui.set_global_completed(idx)
        else:
            progress.update(1)

    # Restore original ordering; joblib yields results as workers finish.
    simulation_outputs.sort(key=lambda item: item[0])

    if ui is not None:
        ui.end_global_progress()
    elif progress is not None:
        progress.close()

    x = np.vstack([output[0] for _, output in simulation_outputs])
    mask = np.vstack([output[1] for _, output in simulation_outputs])
    return x, mask

class PytreeAdapter:
    """Converts between a batch of pytrees and a 2-D (B, D) array."""

    def __init__(self, example_theta: dict[str, jnp.ndarray]):
        self._keys = list(example_theta.keys())
        self._leaf_shapes = [jnp.asarray(example_theta[k]).shape for k in self._keys]
        self._leaf_sizes = [int(np.prod(shape)) if shape else 1 for shape in self._leaf_shapes]
        self._cumulative_sizes = np.cumsum(self._leaf_sizes)
        self._key_slices: dict[str, slice] = {}

        start = 0
        for key, size in zip(self._keys, self._leaf_sizes):
            end = start + size
            self._key_slices[key] = slice(start, end)
            start = end

        def _ravel_single(theta_sample: dict[str, jnp.ndarray]) -> jnp.ndarray:
            flat_leaves = [
                jnp.reshape(jnp.asarray(theta_sample[k]), (size,))
                for k, size in zip(self._keys, self._leaf_sizes)
            ]
            return jnp.concatenate(flat_leaves, axis=0)

        def _unravel_single(flat_vector: jnp.ndarray) -> dict[str, jnp.ndarray]:
            flat_vector = jnp.asarray(flat_vector)
            splits = jnp.split(flat_vector, self._cumulative_sizes[:-1])
            return {
                key: split.reshape(shape)
                for key, shape, split in zip(self._keys, self._leaf_shapes, splits)
            }

        self._ravel_single = _ravel_single
        self.ravel = _ravel_single
        self.unravel = _unravel_single

    def to_array(self, theta_batch_tree: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return jax.vmap(self._ravel_single)(theta_batch_tree)

    def to_pytree(self, X: jnp.ndarray):
        return jax.vmap(self.unravel)(X)

    def flat_slice(self, key: str) -> slice:
        return self._key_slices[key]

    def flat_view(self, theta_array: jnp.ndarray, key: str) -> jnp.ndarray:
        sl = self.flat_slice(key)
        view = theta_array[..., sl]
        shape = self._leaf_shapes[self._keys.index(key)]
        if len(shape) == 0:
            return view.reshape(theta_array.shape[:-1])
        return view.reshape(theta_array.shape[:-1] + shape)

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def key_index(self, key: str) -> int:
        return self._keys.index(key)

def convert_x_in_named_dataset(
        dataset: healpix_map_dataset_idx | healpix_map_dataset,
        adapter: Optional[PytreeAdapter] = None
) -> healpix_map_dataset_idx | healpix_map_dataset:
    fields = dataset._asdict()
    x = fields['x']
    if isinstance(x, dict):
        if adapter is None:
            raise ValueError('PytreeAdapter required to flatten theta dicts.')
        x_tree = {k: jnp.asarray(v) for k, v in x.items()}
        flattened = adapter.to_array(x_tree)
        fields['x'] = np.asarray(flattened)
    else:
        fields['x'] = np.asarray(x)
    return type(dataset)(**fields)

def is_integerish_f32(x, ulps=1):
    x = np.asarray(x, dtype=np.float32)
    # handle zeros cleanly
    mag = np.maximum(np.abs(x), np.float32(1.0))
    m = np.floor(np.log2(mag)).astype(np.int32)
    spacing = np.exp2(m - 23).astype(np.float32)   # 1 ulp at each magnitude
    return np.all(np.abs(x - np.rint(x)) <= ulps * spacing)

class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def save_dict_npz(path, x: dict):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))  # ensure computed
    np.savez_compressed(path, **{k: jax.device_get(v) for k, v in x.items()})

def load_dict_npz(path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: jnp.array(d[k]) for k in d.files}

def spherical_to_cartesian(theta_phi: tuple[NDArray, NDArray]) -> NDArray:
    '''
    Transform spherical coordinates in the form (theta, phi) to Cartesian
    coordinates given r = 1. Theta is the polar angle and phi the azimuthal
    angle. The polar angle runs from 0 to 180 degrees, where zero degrees
    corresponds to z = 1 in Cartesian coordinates. From Honours code.
    '''
    x = np.sin(theta_phi[0]) * np.cos(theta_phi[1])
    y = np.sin(theta_phi[0]) * np.sin(theta_phi[1])
    z = np.cos(theta_phi[0])
    xyz = np.stack([x, y, z])
    return xyz

def np_sph2cart_unitsphere(
        phi: NDArray, 
        theta: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    '''
    Transform spherical coordinates longitude and colatitude in radians to
    Cartesian.
    '''
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def jax_sph2cart(
        phi: jnp.ndarray, 
        theta: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    '''
    Transform spherical coordinates longitude and colatitude in radians to
    Cartesian.
    '''
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return x, y, z

def jax_cart2sph(
        x: jnp.ndarray, 
        y: jnp.ndarray, 
        z: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    norm = jnp.sqrt(jnp.square(x) + jnp.square(y) + jnp.square(z))
    theta = jnp.arccos(z / norm)
    phi = jnp.arctan2(y, x)
    phi[phi < 0] += 2 * jnp.pi
    return phi, theta

def sample_unif(unif: Tensor, low_high: tuple[Tensor, Tensor]) -> Tensor:
    '''
    (b - a) * u + a
    '''
    low = low_high[0]; high = low_high[1]
    return (high - low) * unif + low

def sample_unif_np(unif: NDArray, low: NDArray, high=NDArray) -> NDArray:
    '''
    (b - a) * u + a
    '''
    return (high - low) * unif + low

def unif_pdf(low_high: list[float]) -> float:
    low = low_high[0]; high = low_high[1]
    return 1 / (high - low)

def sample_polar(unif: Tensor, low_high: tuple[Tensor, Tensor]) -> Tensor:
    '''
    :param unif: Uniform deviates on [0, 1).
    :param low_high: Tuple of minimum and maximum polar angle (colatitude, radians).
    :return: Samples on [0, pi).
    '''
    low = low_high[0]; high = low_high[1]
    unif_theta = torch.arccos(torch.cos(low) + unif * (torch.cos(high) - torch.cos(low)))
    return unif_theta

def sample_polar_np(
        rngkey_or_unifs: NPKey | NDArray,
        n_samples: int = 1,
        low: float = -90.,
        high: float = 90.,
        dtype: DTypeLike = np.float32
) -> NDArray:
    if type(rngkey_or_unifs) == NPKey:
        unifs = rngkey_or_unifs.uniform((n_samples,), dtype=dtype)
    else:
        unifs = rngkey_or_unifs

    low_rad = np.pi / 2 - np.deg2rad(low)
    high_rad = np.pi / 2 - np.deg2rad(high)
    unif_theta = np.arccos(
        np.cos(low_rad) + unifs * (np.cos(high_rad) - np.cos(low_rad))
    )
    return 90. - np.rad2deg(unif_theta)

def sample_polar_jax(
        rng_key, 
        minval: float = -90.,
        maxval: float = 90.
) -> jnp.ndarray:
    '''
    :param minval: Degrees latitude.
    :param maxval: Degrees latitude.
    :return: Samples in degrees latitude (-90, 90).
    '''
    unif = jax.random.uniform(rng_key, shape=())
    minval_rad = jnp.pi / 2 - jnp.deg2rad(minval)
    maxval_rad = jnp.pi / 2 - jnp.deg2rad(maxval)
    unif_theta = jnp.arccos(
        jnp.cos(minval_rad) + unif * (jnp.cos(maxval_rad) - jnp.cos(minval_rad))
    )
    return 90. - jnp.rad2deg(unif_theta)

def polar_pdf(theta: float, low_high: list[float]):
    low = low_high[0]; high = low_high[1]
    return - np.sin(theta) / (np.cos(high) - np.cos(low))

def polar_logpdf_np(
        latitude: NDArray,
        low: float = -90.,
        high: float = 90.
) -> NDArray:
    '''
    :param latitude: Latitude in degrees (-90, 90).
    '''
    theta = np.pi / 2 - np.deg2rad(latitude)

    # maxval and minval flip remapping from (-90, 90) to (0, pi) polar
    high_rad = np.pi / 2 - np.deg2rad(low)
    low_rad = np.pi / 2 - np.deg2rad(high)

    return (
        np.log(-np.sin(theta) / (np.cos(high_rad) - np.cos(low_rad)))
      + np.log(np.pi / 180) # p(theta_rad) -> p(theta_deg) Jacobian
    )


def polar_logpdf_jax(
        latitude: jnp.ndarray, 
        minval: float = -90.,
        maxval: float = 90.
) -> jnp.ndarray:
    '''
    :param latitude: Latitude in degrees (-90, 90).
    '''
    theta = jnp.pi / 2 - jnp.deg2rad(latitude)

    # maxval and minval flip remapping from (-90, 90) to (0, pi) polar
    maxval_rad = jnp.pi / 2 - jnp.deg2rad(minval)
    minval_rad = jnp.pi / 2 - jnp.deg2rad(maxval)

    # restrict to prior range
    support = jnp.logical_and(latitude >= minval, latitude <= maxval)
    density = -jnp.sin(theta) / (jnp.cos(maxval_rad) - jnp.cos(minval_rad))
    log_density = jnp.log(jnp.abs(density)) + jnp.log(jnp.pi / 180)

    return jnp.where(support, log_density, -jnp.inf)

def softplus_pos(x):  # avoids zero scales
    return F.softplus(x) + 1e-8

def dipole_signal(Theta, nside=32, device='cpu'):
    Nbar, D, phi, theta = torch.as_tensor(Theta, device=device, dtype=torch.float64)
    pixel_indices = torch.arange(hp.nside2npix(nside))
    pixel_vectors = torch.as_tensor(
        torch.stack(
            hp.pix2vec(nside, pixel_indices, nest=True)
        ),
        device=device
    )
    dipole_vector = D * spherical_to_cartesian((theta, phi), device=device)
    poisson_mean = Nbar * (1 + torch.einsum('i,i...', dipole_vector, pixel_vectors))
    return poisson_mean

def simulation(Theta, nside=32, device='cpu'):
    poisson_mean = dipole_signal(Theta, nside, device)
    return poisson(poisson_mean)

def enforce_batchwise_input(Theta: NDArray, ndim: int) -> NDArray:
        if Theta.shape == (ndim,):
            Theta = Theta.reshape(1, ndim)
        return Theta

def convert_to_l_dash(l):
    '''When plotting the dynesty histogram on top of the healpy projection
    plot, for whatever reason despite using galactic coordinates the healpy
    plot needs the values for l to be between [-pi and pi], not [0 and 2pi].
    This performs the conversion accordingly.
    
    Parameters
    ----------
    l : the azimuthal angle of each point in galactic coordinates in radians,
    moving from 0 to 2pi.'''

    try:
        l_dash = []
        for i in range(0,len(l)):
            if l[i] <= np.pi:
                l_dash.append(-l[i])
            elif l[i] > np.pi:
                l_dash.append(2 * np.pi - l[i])
            else:
                print('No you should not be here!')
        return np.array(l_dash)
    except TypeError:
        if l <= np.pi:
            return -l
        elif l > np.pi:
            return 2 * np.pi - l
        else:
            print('No you should not be here!')

def sigma_to_prob2D(sigma: list) -> Tensor:
    '''Convert sigma significance to mass enclosed inside a 2D normal
    distribution using the explicit formula for a 2D normal.
    :param sigma: the levels of significance
    :returns: the probability enclosed within each significance level'''
    return 1.0 - np.exp(-0.5 * np.asarray(sigma)**2)

def compute_2D_contours(
        P_xy: Tensor,
        contour_levels: list[float]
) -> tuple[Tensor]:
    '''
    Compute contour heights corresponding to sigma levels of probability
    density by creating a mapping (interpolation function) from the CDF
    (enclosed prob) to some arbitrary level of probability density.
    from here: https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution

    :param P_xy: normalised 2D probability (not density)
    :param contour_levels: pass list of sigmas at which to draw the contours
    :return:
        1. vector of probabilities corresponding to heights at which to
        draw the contours (pass to e.g. plt.contour with levels= kwarg);
        2. uniformly spaced probability levels between 0 and max prob
        3. CDF at given P_level, of length 1000 (the hardcoded number of
            P_levels)
    '''
    P_levels = np.linspace(0, P_xy.max(), 1000)
    mask = (P_xy >= P_levels[:, None, None])
    P_integral = (mask * P_xy).sum(axis=(1,2))
    f = interp1d(P_integral, P_levels)
    t_contours = np.flip(f(sigma_to_prob2D(contour_levels)))
    return t_contours, P_levels, P_integral

def samples_to_hpmap(
        phi: Tensor,
        theta: Tensor,
        lonlat: bool = False,
        nside: int = 64,
        smooth: None | float = None
    ) -> Tensor:
    '''
    Turn numerical samples in phi-theta space to a healpy map, in the native
    coords of phi theta, defining the probability of a sample (phi_i, theta_i)
    lying in a given pixel.

    :param phi: vector of phi samples in spherical coordinates: [0, 2pi)
    :param theta: vector of theta samples in spherical coordinates: [0, pi]
    :param lonlat: if True, phi ~ [0, 360] and theta ~ [-90, 90]; else,
        phi ~ [0, 2pi] and theta ~ [0, pi]
    :param weights: weights of each samples, defaults to None
    :param nside: nside (resolution) of binning of nested samples, i.e. the
        resolution of the posterior probability map
    '''
    # note: labels flip where lonlat=True... thanks healpy
    if lonlat:
        sample_pixel_indices = hp.ang2pix(
            nside=nside, theta=phi, phi=theta, lonlat=lonlat
        )
    else:
        sample_pixel_indices = hp.ang2pix(
            nside=nside, theta=theta, phi=phi
        )
    sample_count_map = np.bincount(
        sample_pixel_indices, minlength=hp.nside2npix(nside)
    )
    
    # convert count to prob density
    map_total = np.sum(sample_count_map)
    sample_pdensity_map = sample_count_map / map_total
    
    if smooth is not None:
        # healpy's smooth function (in samples_to_hpmap) works in sph
        # harmonic space, and produces a small number of very small
        # negative values; the sum is also not preserved.
        # correct by replacing negative values with 0 and renormalise.
        smooth_map = hp.sphtfunc.smoothing(sample_pdensity_map, sigma=smooth)
        smooth_map[smooth_map < 0] = 0
        smooth_map /= np.sum(smooth_map)
        return smooth_map
    else:
        return sample_pdensity_map

def omega_to_theta(omega: float) -> np.float64:
    '''
    Convert solid angle in steradins to theta in radians for
    a cone section of a sphere.
    
    :param omega: solid angle in steradians.
    '''
    return np.arccos( 1 - omega / (2 * np.pi) )

def equatorial_to_ecliptic(ra, dec, output_unit='radians'):
    eq = SkyCoord(ra, dec, unit=u.deg)
    ecl = eq.transform_to('barycentricmeanecliptic')
    if output_unit == 'radians':
        return ecl.lon.rad, ecl.lat.rad
    elif output_unit == 'degrees':
        return ecl.lon.deg, ecl.lat.deg
    else:
        raise Exception(
            'Not a valid unit. Select either radians of degrees.')

def group_healpix_children(
        healpix_map: NDArray, 
        super_pixel_nside: int
) -> NDArray:
    '''
    :param healpix_map: Input healpix map, of shape (npix,)
        or (n_batches, n_pix) if passing batchwise.
    return: Input healpix map grouped by child pixels, of original_shape
        (super_npix, n_children) if no batches or
        (n_batches, super_npix, n_children) if passing batchwise.
    '''
    if len(healpix_map.shape) == 1:
        n_batches: int = 0
    else:
        n_batches: int = healpix_map.shape[0]

    input_npix: int = healpix_map.shape[-1]
    input_nside: int = hp.npix2nside(input_npix)
    output_npix: int = hp.nside2npix(super_pixel_nside)
    total_possible_levels: np.int64 = np.log2(input_nside).astype(np.int64)
    all_levels = []

    n = input_nside
    for _ in range(total_possible_levels):
        all_levels.append(n)
        n //= 2

    assert super_pixel_nside in all_levels, (
        'Super pixel nside not valid (must be larger an input nside).'
    )
    # # get indices of parents for each child through bit shifts
    # n_right_bit_shifts = 2 * coarse_order
    # parent_indices = input_ipix >> n_right_bit_shifts

    # get number of downscales to reach desired coarse nside
    coarse_order: int = (
        np.log2(input_nside) - np.log2(super_pixel_nside)
    ).astype(int)
    n_children: int = 4 ** coarse_order
    if n_batches == 0:
        return healpix_map.reshape(output_npix, n_children)
    else:
        return healpix_map.reshape(n_batches, output_npix, n_children)

class ParameterMap:
    def __init__(self,
            pixel_indices: NDArray[np.int_],
            parameter: NDArray[np.float64],
            nside: int
    ) -> None:
        self.parameter_dict = defaultdict(list)
        self.nside = nside
        for idx, pix in enumerate(pixel_indices):
            self.parameter_dict[int(pix)].append( parameter[idx] )
        self.median_map = None

    def get_map(self) -> NDArray[np.float64]:
        if self.median_map is None:
            n_pix = hp.nside2npix(self.nside)
            self.median_map = np.full(n_pix, np.nan)
            
            for pix, parameter_values in self.parameter_dict.items():
                if len(parameter_values) > 0:
                    self.median_map[pix] = np.median(parameter_values)
        
        return self.median_map

class Sample1DHistogram:
    def __init__(self) -> None:
        pass

    def build(self,
            x_data = None,
            **hist_kwargs
        ) -> None:
        counts, x_edges = np.histogram(x_data, **hist_kwargs)
        x_centres = (x_edges[:-1] + x_edges[1:]) / 2
        pdf = counts / np.sum(counts)
        cdf = np.cumsum(pdf)
        self.inverse_cdf = lambda x: np.interp(
            x,
            cdf,
            x_centres
        )
    
    def save_data(self, save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        sampler_data = self.inverse_cdf
        
        with open(f'{save_dir}sampler_data.pkl', 'wb') as handle:
            pickle.dump(sampler_data, handle)

    def load_data(self, save_dir: str) -> None:
        with open(f'{save_dir}sampler_data.pkl', 'rb') as handle:
            data = pickle.load(handle)
        
        self.inverse_cdf = data
    
    def sample(self, n_samples: int) -> np.ndarray:
        uniform_deviate = np.random.uniform(0, 1, n_samples)
        samples = self.inverse_cdf(uniform_deviate)
        return samples

class Sample2DHistogram:
    def __init__(self) -> None:
        pass

    def build(self,
            x_data = None,
            y_data = None,
            **hist_kwargs
        ) -> None:

        counts_2d, self.x_edges, self.y_edges = np.histogram2d(
            x_data, y_data, **hist_kwargs
        )
        
        x_centres = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        y_centres = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        
        pdf_2d = counts_2d / np.sum(counts_2d)
        pdf_x = np.sum(pdf_2d, axis=1)

        # Filter out x-bins with zero marginal probability
        valid_x_mask = pdf_x > 0
        self.x_centres = x_centres[valid_x_mask]
        self.x_edges = x_centres[valid_x_mask]
        pdf_x_valid = pdf_x[valid_x_mask]
        pdf_2d_valid = pdf_2d[valid_x_mask, :]

        cdf_x = np.cumsum(pdf_x_valid)
        assert np.isclose(cdf_x[-1], 1.)
        
        self.inverse_cdf_x = lambda x: np.interp(
            x,
            cdf_x,
            self.x_centres
        )

        self.cdf_y_lookup = {}
        for i in range(0, len(self.x_centres)):
            y_row = pdf_2d_valid[i, :] / np.sum(pdf_2d_valid[i, :])
            cdf_y = np.cumsum(y_row)
            assert np.isclose(cdf_y[-1], 1.)
            
            # for some reason I need to pass cdf_y and y_centres as default
            # kwargs, otherwise they don't change on each for loop
            inverse_cdf_y = (
                lambda y, cdf_y=cdf_y, y_centres=y_centres: np.interp(
                    y,
                    cdf_y,
                    y_centres
                )
            )
            self.cdf_y_lookup[i] = inverse_cdf_y
    
    def save_data(self, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        sampler_data = (self.inverse_cdf_x, self.x_edges, self.cdf_y_lookup)
        
        with open(f'{save_dir}sampler_data.pkl', 'wb') as handle:
            pickle.dump(sampler_data, handle)
    
    def load_data(self, save_dir: str) -> None:
        with open(f'{save_dir}sampler_data.pkl', 'rb') as handle:
            data = pickle.load(handle)
        
        self.inverse_cdf_x, self.x_edges, self.cdf_y_lookup = data

    def sample(self, n_samples: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # timing: n_samples = 1_000_000
        uniform_deviate_x = np.random.uniform(0, 1, n_samples)
        uniform_deviate_y = np.random.uniform(0, 1, n_samples)

        x_samples = self.inverse_cdf_x(uniform_deviate_x) # 30 ms
        y_cdf_indices = np.searchsorted(self.x_edges, x_samples) # 25 ms
        y_cdfs = itemgetter(*y_cdf_indices)(self.cdf_y_lookup) # 37 ms

        y_samples = np.asarray(
            [cdf_y(u_y) for cdf_y, u_y in zip(y_cdfs, uniform_deviate_y)]
        ) # 790 ms
        
        return x_samples, y_samples


class MultinomialSample2DHistogram:
    """
    Fast 2D histogram sampling using multinomial distribution.
    
    This approach treats the 2D histogram as a single multinomial distribution
    over all bins, allowing for very fast sampling by directly using np.random.choice.
    Expected to be ~10-20x faster than the conditional CDF approach.
    """
    
    def __init__(self) -> None:
        pass

    def build(self,
            x_data = None,
            y_data = None,
            **hist_kwargs
        ) -> None:
        """
        Build the multinomial sampler from data with jittering support.
        
        Parameters:
        -----------
        x_data : array-like
            X coordinates of data points
        y_data : array-like  
            Y coordinates of data points
        **hist_kwargs : dict
            Additional arguments passed to np.histogram2d
        """
        
        # Create 2D histogram
        counts_2d, self.x_edges, self.y_edges = np.histogram2d(
            x_data, y_data, **hist_kwargs
        )
        
        # Calculate bin centers
        x_centres = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        y_centres = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        
        # Calculate bin widths for jittering
        self.x_bin_widths = np.diff(self.x_edges)
        self.y_bin_widths = np.diff(self.y_edges)
        
        # Create 2D coordinate grids for centers and widths
        self.x_centres_2d, self.y_centres_2d = np.meshgrid(
            x_centres, y_centres, indexing='ij'
        )
        x_widths_2d, y_widths_2d = np.meshgrid(
            self.x_bin_widths, self.y_bin_widths, indexing='ij'
        )
        
        # Flatten coordinate grids for multinomial sampling
        self.x_flat = self.x_centres_2d.flatten()
        self.y_flat = self.y_centres_2d.flatten()
        self.x_widths_flat = x_widths_2d.flatten()
        self.y_widths_flat = y_widths_2d.flatten()
        
        # Flatten counts and normalize to probabilities
        counts_flat = counts_2d.flatten()
        self.probs_flat = counts_flat / np.sum(counts_flat)
        
        # Store original shape for potential debugging
        self.original_shape = counts_2d.shape
        
        # Filter out zero-probability bins for efficiency (optional)
        nonzero_mask = self.probs_flat > 0
        if np.sum(nonzero_mask) < len(self.probs_flat):
            self.x_flat = self.x_flat[nonzero_mask]
            self.y_flat = self.y_flat[nonzero_mask]
            self.x_widths_flat = self.x_widths_flat[nonzero_mask]
            self.y_widths_flat = self.y_widths_flat[nonzero_mask]
            self.probs_flat = self.probs_flat[nonzero_mask]
            # Renormalize after filtering
            self.probs_flat = self.probs_flat / np.sum(self.probs_flat)
        
        print(f"MultinomialSample2DHistogram built with {len(self.probs_flat)} active bins")

    def save_data(self, save_dir: str) -> None:
        """Save the multinomial sampler data."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        sampler_data = {
            'x_flat': self.x_flat,
            'y_flat': self.y_flat,
            'x_widths_flat': self.x_widths_flat,
            'y_widths_flat': self.y_widths_flat,
            'probs_flat': self.probs_flat,
            'x_edges': self.x_edges,
            'y_edges': self.y_edges,
            'original_shape': self.original_shape
        }
        
        with open(f'{save_dir}multinomial_sampler_data.pkl', 'wb') as handle:
            pickle.dump(sampler_data, handle)

    def load_data(self, save_dir: str) -> None:
        """Load the multinomial sampler data."""
        with open(f'{save_dir}multinomial_sampler_data.pkl', 'rb') as handle:
            sampler_data = pickle.load(handle)
        
        self.x_flat = sampler_data['x_flat']
        self.y_flat = sampler_data['y_flat']
        self.x_widths_flat = sampler_data['x_widths_flat']
        self.y_widths_flat = sampler_data['y_widths_flat']
        self.probs_flat = sampler_data['probs_flat']
        self.x_edges = sampler_data['x_edges']
        self.y_edges = sampler_data['y_edges']
        self.original_shape = sampler_data['original_shape']

    def sample(
            self,
            n_samples: int,
            rng: Optional[np.random.Generator] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Sample from the 2D distribution using multinomial sampling with uniform jittering.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        x_samples : NDArray[np.float64]
            X coordinates of samples with uniform jittering within bins
        y_samples : NDArray[np.float64]
            Y coordinates of samples with uniform jittering within bins
        """
        if rng is None:
            rng = np.random.default_rng()

        # Multinomial sampling to select bins
        indices = rng.choice(
            len(self.probs_flat), 
            size=n_samples, 
            p=self.probs_flat
        )
        
        # Get bin centers and widths for selected bins
        x_centers = self.x_flat[indices]
        y_centers = self.y_flat[indices]
        x_widths = self.x_widths_flat[indices]
        y_widths = self.y_widths_flat[indices]
        
        # Add uniform jitter within each bin
        # Jitter is uniform in [-width/2, +width/2] around bin center
        x_jitter = rng.uniform(-0.5, 0.5, n_samples) * x_widths
        y_jitter = rng.uniform(-0.5, 0.5, n_samples) * y_widths
        
        # Apply jittering to get continuous samples
        x_samples = x_centers + x_jitter
        y_samples = y_centers + y_jitter
        
        return x_samples, y_samples
    
    def get_bin_info(self) -> dict:
        """
        Get information about the binning for debugging/analysis.
        
        Returns:
        --------
        info : dict
            Dictionary containing bin information
        """
        return {
            'n_bins_total': len(self.x_flat),
            'x_range': (self.x_edges[0], self.x_edges[-1]),
            'y_range': (self.y_edges[0], self.y_edges[-1]),
            'original_shape': self.original_shape,
            'min_probability': np.min(self.probs_flat),
            'max_probability': np.max(self.probs_flat)
        }

class Samples:
    def __init__(self, samples: Tensor | NDArray) -> None:
        if type(samples) is NDArray:
            self.samples = torch.as_tensor(samples)
        else:
            self.samples = samples
        self.total_samples = samples.shape[0]

    def sample(self, num_simulations: tuple[int]) -> Tensor:
        '''
        :param num_simulations: e.g. (5,).
        '''
        indices = torch.as_tensor(
            np.random.choice(
                self.total_samples,
                size=num_simulations[0],
                replace=True
            )
        )
        return self.samples[indices]
