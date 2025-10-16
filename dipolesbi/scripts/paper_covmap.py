import os
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.configs import CatwiseConfig
from dipolesbi.tools.utils import ParameterMap
from dipolesbi.tools.plotting import smooth_map


mpl.rcParams['text.usetex'] = True

COLORBAR_FONTSIZE = {
    'cbar_label': 20,
    'cbar_tick_label': 18
}


config = CatwiseConfig(
    cat_w1_max=17.0, 
    cat_w12_min=0.5,
    magnitude_error_dist='gaussian'
)
catwise = Catwise(config)
catwise.determine_masked_pixels()
catwise.make_real_sample()

pixel_indices = hp.ang2pix(
    64,
    catwise.real_catalogue['l'], 
    catwise.real_catalogue['b'],
    nest=True,
    lonlat=True
)
coverage_parameter = catwise.real_catalogue['w1cov']
coverage_map = ParameterMap(pixel_indices, coverage_parameter, nside=64).get_map()
coverage_map[~catwise.binary_mask] = np.nan

error_fraction = np.divide(
    catwise.real_catalogue['w1e'],
    catwise.real_catalogue['w1'],
    out=np.zeros_like(catwise.real_catalogue['w1e']),
    where=catwise.real_catalogue['w1'] != 0
) * 100.0
error_map = ParameterMap(pixel_indices, error_fraction, nside=64).get_map()
error_map[~catwise.binary_mask] = np.nan

plt.figure()
hp.projview(
    coverage_map,
    nest=True,
    norm='log',
    unit='W1 Coverage',
    format='%.3g',
    cmap='magma',
    fontsize=COLORBAR_FONTSIZE
)
figure_dir = os.path.join(
    os.path.expanduser('~'),
    'Documents',
    'papers',
    'catwise_sbi',
    'figures'
)
os.makedirs(figure_dir, exist_ok=True)
coverage_figure_path = os.path.join(figure_dir, 'catwise_coverage_map.pdf')
print(f'Saving coverage plot to {coverage_figure_path}...')
plt.savefig(
    coverage_figure_path,
    bbox_inches='tight'
)
plt.close()

plt.figure()
hp.projview(
    error_map,
    nest=True,
    unit=r'W1 Error (\%)',
    format='%.3g',
    cmap='magma',
    fontsize=COLORBAR_FONTSIZE
)
error_figure_path = os.path.join(figure_dir, 'catwise_error_map.pdf')
print(f'Saving error plot to {error_figure_path}...')
plt.savefig(
    error_figure_path,
    bbox_inches='tight'
)
plt.close()

plt.figure()
smooth_map(
    catwise.real_density_map,
    cmap='magma',
    unit='Averaged source count',
    format='%.3g',
    fontsize=COLORBAR_FONTSIZE
)
figure_path = os.path.join(figure_dir, 'catwise_density_map.pdf')
print(f'Saving density plot to {figure_path}...')
plt.savefig(figure_path, bbox_inches='tight')
plt.close()
