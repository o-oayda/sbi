from dipolesbi.tools.maps import SkyMap
from dipolesbi.tools.constants import (
    CMB_PHI_EQ, CMB_THETA_EQ, CMB_L, CMB_B, CMB_PHI_GAL, CMB_THETA_GAL
)
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from dipolesbi.tools.plotting import smooth_map
from dipolesbi.tools.inference import Inference
from corner import corner
from astropy.coordinates import SkyCoord

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})

N_INITIAL = 10_000_000
sim = SkyMap()
sim.generate_dipole_from_base(
    observer_direction=(CMB_PHI_EQ, CMB_THETA_EQ),
    n_initial_points=N_INITIAL,
    flux_percentage_noise='ecliptic',
    minimum_flux_cut=3
)
sim.mask_pixels(fill_value=0)
smooth_map(
    sim.density_map,
    coord=['C', 'G'],
    unit=r'Smoothed source density ($N_{\mathrm{side}} = 32$)'
)
print(np.sum(sim.density_map.numpy()))
plt.scatter(
    x=CMB_PHI_GAL - np.pi,
    y=CMB_THETA_GAL,
    marker='*',
    s=100,
    c='black'
)
plt.savefig('plots/ecliptic_noise_dmap.pdf', bbox_inches='tight')
plt.show()

inferer = Inference()
inferer.load_posterior('based_posterior_sim_ecliptic.pkl')
samples = inferer.sample_amortized_posterior(x_obs=sim.density_map)

# transform samples
samples[:, -2] = np.rad2deg(samples[:, -2])
samples[:, -1] = np.rad2deg(np.pi / 2 - samples[:, -1])
samples[:, -3] = samples[:, -3] / 0.00123
coord = SkyCoord(samples[:, -2], samples[:, -1], unit='deg', frame='icrs')
coord2 = coord.transform_to(frame='galactic')
samples[:, -2] = coord2.l.value
samples[:, -1] = coord2.b.value

corner(
    samples,
    labels=[
        r'$N_{\mathrm{init.}}$',
        r'$v_{\mathrm{obs.}} / v_{\mathrm{CMB}}$',
        r'$l$ ($^\circ$)',
        r'$b$ ($^\circ$)'
    ],
    label_kwargs={
        'size': 20
    },
    truths=[
        N_INITIAL,
        1,
        CMB_L,
        CMB_B
    ],
    truth_color='cornflowerblue'
)
plt.savefig('plots/ecliptic_noise_sbi_corner.pdf', bbox_inches='tight')
plt.show()