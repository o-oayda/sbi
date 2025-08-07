from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.inference import LikelihoodFreeInferer
import numpy as np
from corner import corner
import matplotlib.pyplot as plt
import torch


MEAN_DENSITY = 50.
OBSERVER_SPEED = 2.
DIPOLE_LONGITUDE = 215.
DIPOLE_LATITUDE = 40.
theta0 = np.asarray([MEAN_DENSITY, OBSERVER_SPEED, DIPOLE_LONGITUDE, DIPOLE_LATITUDE]).reshape(4, 1)

model = SimpleDipoleMap()
prior = DipolePrior(mean_count_range=[0.9*MEAN_DENSITY, 1.1*MEAN_DENSITY])
prior.change_kwarg('N', 'mean_density')
simulator = Simulator(prior, model.generate_dipole)
theta, x = simulator.make_batch_simulations(
    n_simulations=50_000, 
    n_workers=32,
    simulation_batch_size=100
)

x0 = model.generate_dipole(*theta0)

inferer = LikelihoodFreeInferer(simulator)
inferer.run_healpix_sbi(load_simulations_in_vram=False)
inferer.plot_loss_curve()
samples = inferer.sample_amortized_posterior(
    torch.as_tensor(x0, device='cuda'), 
    n_samps=100_000
)

corner(samples.numpy(), truths=theta0.flatten(), labels=prior.prior_names)
plt.show()
