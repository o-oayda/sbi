from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.models import CustomModel, DipolePoisson
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.inference import LikelihoodBasedInferer, LikelihoodFreeInferer
import numpy as np
from corner import corner
import matplotlib.pyplot as plt
import torch
import healpy as hp
import harmonic as hm


MEAN_DENSITY = 10_000.
OBSERVER_SPEED = 2.
DIPOLE_LONGITUDE = 215.
DIPOLE_LATITUDE = 40.
theta0 = np.asarray(
    [MEAN_DENSITY, OBSERVER_SPEED, DIPOLE_LONGITUDE, DIPOLE_LATITUDE]
).reshape(4, 1)

model = SimpleDipoleMap(nside=4)
prior = DipolePrior(mean_count_range=[0.9*MEAN_DENSITY, 1.1*MEAN_DENSITY])
prior.change_kwarg('N', 'mean_density')
simulator = Simulator(prior, model.generate_dipole)
# theta, x = simulator.make_batch_simulations(
#     n_simulations=20_000, 
#     n_workers=32,
#     simulation_batch_size=100
# )

x0 = model.generate_dipole(*theta0)
hp.projview(x0.flatten(), nest=True)
plt.show()

inferer = LikelihoodFreeInferer()
inferer.load_posterior('quicktest_NLE_posterior.pkl')
# inferer.run_healpix_sbi(
#     estimator_type='NLE', 
#     load_simulations_in_vram=False, 
#     training_device='cpu'
# )
inferer.posterior.to('cpu')
# inferer.plot_loss_curve()
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

# compute marginal likelihood using NS
custom_model = CustomModel(prior, inferer.posterior.potential)
sbased_inferer = LikelihoodBasedInferer(x0, custom_model)
sbased_inferer.run_ultranest()

classic_model = DipolePoisson(prior, nside=4)
classic_inferer = LikelihoodBasedInferer(x0, classic_model)
classic_inferer.run_ultranest()

print(f'Simulation-based log Z: {sbased_inferer.log_bayesian_evidence}')
print(f'Classic log Z: {classic_inferer.log_bayesian_evidence}')

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
#     chains, training_proportion=0.5
# )
#
# # model = hm.model_legacy.KernelDensityEstimate(ndim, [])
# model = hm.model.RQSplineModel(ndim)
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
