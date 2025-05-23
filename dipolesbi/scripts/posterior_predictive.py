import torch
from dipolesbi.tools.plotting import posterior_predictive_check
from dipolesbi.tools.models import SkyMap
from dipolesbi.tools.inference import Inference
from corner import corner
import matplotlib.pyplot as plt

inferer = Inference()
inferer.load_posterior(file_path='poisson_posterior.pkl')

model = SkyMap()
model.generate_dipole(torch.as_tensor([50, 0.1, 1, 1]))
model.mask_pixels()

samples = inferer.sample_amortized_posterior(x_obs=model.density_map)
corner(samples)
plt.show()

def model_wrapper(Theta):
    model.generate_dipole(Theta)
    return model._density_map

posterior_predictive_check(
    torch.as_tensor(samples),
    model=model_wrapper
)