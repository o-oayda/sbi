from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.inference import LikelihoodFreeInferer
import numpy as np
from corner import corner
import matplotlib.pyplot as plt
import torch
import harmonic as hm


MEAN_DENSITY = 50.
OBSERVER_SPEED = 2.
DIPOLE_LONGITUDE = 215.
DIPOLE_LATITUDE = 40.
theta0 = np.asarray(
    [MEAN_DENSITY, OBSERVER_SPEED, DIPOLE_LONGITUDE, DIPOLE_LATITUDE]
).reshape(4, 1)

model = SimpleDipoleMap()
prior = DipolePrior(mean_count_range=[0.9*MEAN_DENSITY, 1.1*MEAN_DENSITY])
prior.change_kwarg('N', 'mean_density')
simulator = Simulator(prior, model.generate_dipole)
theta, x = simulator.make_batch_simulations(
    n_simulations=20_000, 
    n_workers=32,
    simulation_batch_size=100
)

x0 = model.generate_dipole(*theta0)

inferer = LikelihoodFreeInferer(simulator)
# inferer.load_posterior('quicktest_NPE_posterior.pkl')
inferer.run_healpix_sbi(
    estimator_type='NRE', 
    load_simulations_in_vram=False, 
    training_device='cuda'
)
inferer.posterior.to('cpu')
inferer.plot_loss_curve()
samples = inferer.sample_amortized_posterior(
    torch.as_tensor(x0),
    n_samps=10_000
)

corner(
    samples.detach().cpu().numpy(), 
    truths=theta0.flatten(), 
    labels=prior.prior_names
)
plt.show()

# ndim = samples.shape[-1]
# chains = hm.Chains(ndim)
# logp = inferer.posterior.log_prob(
#     samples,
#     torch.as_tensor(x0)
# ).numpy().astype('float64')
# samples = samples.numpy().astype('float64')
# # chains.add_chains_2d(samples, logp, nchains_in=20)
# chains.add_chain(samples, logp)
# chains.split_into_blocks()
#
# chains_train, chains_infer = hm.utils.split_data(
#     chains, training_proportion=0.9
# )
#
# # model = hm.model_legacy.KernelDensityEstimate(ndim, [])
# model = hm.model.RQSplineModel(ndim, learning_rate=1e-4, n_layers=2, standardize=True, temperature=0.7)
# model.fit(chains_train.samples, epochs=100) 
# # model.fit(chains_train.samples, chains_train.ln_posterior)
# plt.plot(model.loss_values)
# plt.show()
#
# samples = samples.reshape((-1, ndim))
# samp_num = samples.shape[0]
# flow_samples = model.sample(samp_num)
# hm.utils.plot_getdist_compare(samples, flow_samples)
# plt.show()
#
# # Instantiate harmonic's evidence class
# ev = hm.Evidence(chains_infer.nchains, model)
#
# # Pass the evidence class the inference chains and compute the evidence!
# ev.add_chains(chains_infer)
# ln_inv_evidence = ev.ln_evidence_inv
# err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()
# print(f'ln Z = {ln_inv_evidence} +/- {err_ln_inv_evidence}')
