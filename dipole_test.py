# %%
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from funcs import simulation, PolarPrior
from sbi.inference import NPE, NLE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.utils import BoxUniform
import torch
from sbi.analysis import pairplot
from corner import corner
# %%

D = 0.5
PHI = 3
THETA = 1

dmap = simulation([D, PHI, THETA])
hp.projview(dmap)
plt.show()
# %%
prior_theta = BoxUniform(
    low=0 * torch.ones(1), high=torch.pi * torch.ones(1)
)
# prior_theta = PolarPrior(
    # theta_low=0 * torch.ones(1), theta_high=torch.pi * torch.ones(1)
# )
prior_d = BoxUniform(
    low=0 * torch.ones(1), high=1 * torch.ones(1)
)
prior_phi = BoxUniform(
    low=0 * torch.ones(1), high=2 * torch.pi * torch.ones(1)
)

prior, num_parameters, prior_returns_numpy = process_prior(
    [prior_d, prior_phi, prior_theta],
)

simulator = process_simulator(simulation, prior, prior_returns_numpy)
check_sbi_inputs(simulator, prior)
inference = NPE(prior=prior)

n_workers = 32
n_simulations = 2000
# theta, x = simulate_for_sbi(
    # simulator, proposal=prior, num_simulations=n_simulations, num_workers=n_workers,
    # show_progress_bar=True
# )
# %%
# theta = prior.sample((num_simulations,))
# x = simulator(theta)
# print("theta.shape", theta.shape)
# print("x.shape", x.shape)

n_rounds = 3
proposal = prior
for _ in range(n_rounds):
    theta, x = simulate_for_sbi(
        simulator,
        proposal=prior, num_simulations=n_simulations, num_workers=n_workers,
        show_progress_bar=True
    )
    density_estimator = inference.append_simulations(
        theta, x, proposal=proposal
    ).train()
    posterior = inference.build_posterior(density_estimator)
    proposal = posterior.set_default_x(dmap)

# inference = inference.append_simulations(theta, x)
# density_estimator = inference.train()
# posterior = inference.build_posterior(density_estimator)

print(posterior)

samples = posterior.sample((10000,), x=dmap)

corner(samples.cpu().detach().numpy(), truths=[D, PHI, THETA])
# _ = pairplot(samples,
#              limits=[[-2, 2], [-2, 2], [-2, 2]],
#              figsize=(6, 6),
#              labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"])
plt.show()
# %%
