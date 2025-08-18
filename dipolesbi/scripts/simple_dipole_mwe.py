from typing import Optional
from scipy.stats import norm
from torch.types import Tensor
from dipolesbi.scripts.healpix_transform1 import HealpixHaarPyramid
from dipolesbi.scripts.healpix_transform2 import HealpixSOPyramid, learn_transformation
from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.models import CustomModel, DipolePoisson
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.inference import LikelihoodBasedInferer, LikelihoodFreeInferer
from dipolesbi.scripts.transform_test import LogAffineTransform
import numpy as np
import matplotlib.pyplot as plt
import torch
import healpy as hp
from getdist import plots, MCSamples
from tabulate import tabulate
from dipolesbi.tools.transforms import AnscombeTransform

# TODO
# - compare with no Affine transform (sbi native structured)
# - simulation spam?

LOAD_PATH_RAW: Optional[str] = None # 'raw_NLE_nside8.pkl'
LOAD_PATH_TRANS: Optional[str] = None # 'trans_NLE_nside8.pkl'
TRAINING_DEVICE: str = 'mps'
N_WORKERS: int = 12

## ---------------------------------------------------
## MODEL PARAMETERS
# nside 4 ok
# nside 16 we start to see very significant discrepancy with log z
# transforming the data seems to make no difference or is slightly worse
NSIDE = 8
NPIX = hp.nside2npix(NSIDE)
TOTAL_SOURCES = 1_920_000
MEAN_DENSITY = TOTAL_SOURCES / hp.nside2npix(NSIDE)
OBSERVER_SPEED = 2.
DIPOLE_LONGITUDE = 215.
DIPOLE_LATITUDE = 40.
theta0 = np.asarray(
    [MEAN_DENSITY, OBSERVER_SPEED, DIPOLE_LONGITUDE, DIPOLE_LATITUDE]
).reshape(4, 1)

## SKYMAP PARAMETERS
EQUATOR_MASK = 0.

## PRIOR
mean_count_range = [0.95*MEAN_DENSITY, 1.05*MEAN_DENSITY]
prior = DipolePrior(mean_count_range=mean_count_range)
prior.change_kwarg('N', 'mean_density')
## ---------------------------------------------------

# SIMULATE DATA
model = SimpleDipoleMap(nside=NSIDE)
if EQUATOR_MASK != 0.:
    model.equatorial_plane_mask(EQUATOR_MASK)
simulator = Simulator(prior, model.generate_dipole)
theta, x = simulator.make_batch_simulations(
    n_simulations=50_000, 
    n_workers=N_WORKERS,
    simulation_batch_size=100
)

## AFFINE TRANSFORM
# Average over all samples for batch std.
logx = torch.log1p(x)
mu_logx = torch.mean(logx)
sample_std = torch.std(logx, dim=1)
sample_std[sample_std < 1e-14] = 1e-14
t_std_logx = torch.mean(sample_std)

ans_x = 2 * torch.sqrt(x + 0.375)
mu_ansx = torch.mean(ans_x)
sample_std = torch.std(ans_x, dim=1)
sample_std[sample_std < 1e-14] = 1e-14
t_std_ansx = torch.mean(sample_std)

mu = torch.mean(x)
sample_std = torch.std(x, dim=1)
sample_std[sample_std < 1e-14] = 1e-14
t_std = torch.mean(sample_std)

# transform = LogAffineTransform(mu, t_std, learn_mu=False) 
# transform = AnscombeTransform() # a bit shitter than LogAffineTransform
# transform = HealpixHaarPyramid(nside_fine=NSIDE)
transform = HealpixSOPyramid(NSIDE)
normalise = lambda input: (input - mu) / t_std
unnormalise = lambda input: t_std * input + mu
# normalise = lambda input: (2 * torch.sqrt(input + 0.375) - mu_ansx) / t_std_ansx
# unnormalise = lambda input: (
#     0.125 * (
#         2 * mu_ansx**2
#       + 2 * t_std_ansx**2 * input**2
#       + 4 * mu_ansx * t_std_ansx * input
#       - 3
#     )
# )
x_normalised = normalise(x)
transform, history, validation = learn_transformation(
    transform, 
    data=x_normalised, 
    epochs=10
)
# losses = pretrain(transform, data=(x - mu) / t_std, steps=500, batch_size=64, lr=1e-3)
plt.plot(history['train'])
plt.title('Training loss')
plt.show()
plt.plot(history['val'])
plt.title('Validation loss')
plt.show()

# CHECK RECONSTRUCTION ERROR
# D = simulator.x
# D_norm = ( D - mu ) / t_std
z, per_level_coeffs, _, totlogdet = transform.forward(x_normalised)
z = z.detach()
x_rec_norm, itotlogdet = transform.inverse(z)
x_rec = unnormalise(x_rec_norm)
# D_rec = D_norm_rec * t_std + mu
recon_err = (x_rec - x).abs().max().item()
print(f"Max |reconstruction error|: {recon_err:.3e}")

# INSPECT COEFFS
for i, (n_parents, coeffs) in enumerate(zip(transform.parents_at_levels, per_level_coeffs)):
    coarse_coeffs = coeffs[:, :, 0].detach().flatten()
    detail1_coeffs = coeffs[:, :, 1].detach().flatten()
    mu_coarse = transform.mu_list[i][0].detach()
    sigma_coarse = torch.exp(transform.log_sigma_list[i][0]).detach()
    mu_detail1 = transform.mu_list[i][1].detach()
    sigma_detail1 = torch.exp(transform.log_sigma_list[i][1]).detach()
    
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(coarse_coeffs, bins=200, color='tab:blue', alpha=0.3, density=True)
    axs[0].hist(coarse_coeffs, bins=200, color='tab:blue', density=True, histtype='step', label='Coarse')
    axs[1].hist(detail1_coeffs, bins=200, color='tab:orange', alpha=0.3, density=True)
    axs[1].hist(detail1_coeffs, bins=200, color='tab:orange', density=True, histtype='step', label='Detail 1')

    x = np.linspace(-5, 5, 10000)
    y_coarse = norm.pdf(x, loc=mu_coarse, scale=sigma_coarse)
    axs[0].plot(x, y_coarse, c='tab:red', label='Normal (coarse fit)')
    y_fine = norm.pdf(x, loc=mu_detail1, scale=sigma_detail1)
    axs[1].plot(x, y_fine, c='tab:red', label='Normal (detail1 fit)')

    fig.legend()
    plt.show()

# MOCK TRUE SAMPLE
x0 = model.generate_dipole(*theta0)
x0_flat = x0.flatten()
mask_map = ~np.isnan(x0_flat)
x0_truncated = x0_flat[mask_map]
x0_truncated_batchwise = x0_truncated.reshape(1, len(x0_truncated))
z0_truncated, *_ = transform.forward(normalise(torch.as_tensor(x0_truncated_batchwise)))
z0_truncated = z0_truncated.detach().numpy()

hp.projview(x0_flat, nest=True)
plt.show()
hp.projview(z0_truncated.squeeze(), nest=True)
plt.show()

# TRAIN NLEs
raw_data_simulator = simulator
transformed_simulator = Simulator(prior, model.generate_dipole)
transformed_simulator.x = z; transformed_simulator.theta = theta

raw_inferer = LikelihoodFreeInferer(simulator)
if LOAD_PATH_RAW is not None:
    raw_inferer.load_posterior(LOAD_PATH_RAW)
else:
# inferer.load_posterior('quicktest_NLE_posterior.pkl')
    # Cut some x
    raw_inferer.simulator.x = raw_inferer.simulator.x[:3000] # type: ignore
    raw_inferer.simulator.theta = raw_inferer.simulator.theta[:3000] # type: ignore
    raw_inferer.run_healpix_sbi(
        estimator_type='NLE', 
        load_simulations_in_vram=False, 
        training_device=TRAINING_DEVICE,
        nan_handle_method='truncate',
        z_score_x='structured',
        n_rounds=3,
        x0_multi=torch.as_tensor(x0_truncated),
        multiround_workers=N_WORKERS
    )
    raw_inferer.posterior.to('cpu') # type: ignore
    raw_inferer.plot_loss_curve()

transformed_inferer = LikelihoodFreeInferer(transformed_simulator)
if LOAD_PATH_TRANS is not None:
    transformed_inferer.load_posterior(LOAD_PATH_TRANS)
else:
    transformed_inferer.run_healpix_sbi(
        estimator_type='NLE', 
        load_simulations_in_vram=False, 
        training_device=TRAINING_DEVICE,
        nan_handle_method='truncate',
        z_score_x=None,
        flow_type='nsf'
    )
    transformed_inferer.posterior.to('cpu') # type: ignore
    transformed_inferer.plot_loss_curve()

raw_samples = raw_inferer.sample_amortized_posterior(
    torch.as_tensor(x0_truncated),
    n_samps=10_000
)
transformed_samples = transformed_inferer.sample_amortized_posterior(
    torch.as_tensor(z0_truncated),
    n_samps=10_000
)

def raw_lnlike(theta: Tensor, x: Tensor) -> Tensor:
    return raw_inferer.posterior.potential(theta, x) - prior.log_prob(theta)

def transformed_lnlike(theta: Tensor, z: Tensor) -> Tensor:
    '''
    SBI NLE returns unnormalised posterior = L(theta) * pi(theta), so subtract
    off the prior term.
    '''
    x_norm, logabsdet = transform.inverse(z)
    x_original = unnormalise(x_norm).detach()
    # -ve sign for log abs det
    return (
        transformed_inferer.posterior.potential(theta, z)
      - prior.log_prob(theta)
      - logabsdet.detach()
      - (torch.log(t_std) * torch.ones(z.shape[-1])).sum(dim=-1) # from normalisation
      # - 0.5 * torch.log(x_original + 0.375).sum(dim=-1) # from Anscombe
    )

# compute marginal likelihood using NS
prior.to('cpu')
raw_custom_model = CustomModel(prior, raw_lnlike)
raw_sbased_inferer = LikelihoodBasedInferer(x0_truncated, raw_custom_model)
raw_sbased_inferer.run_ultranest()

transformed_custom_model = CustomModel(prior, transformed_lnlike)
transformed_sbased_inferer = LikelihoodBasedInferer(z0_truncated, transformed_custom_model)
transformed_sbased_inferer.run_ultranest()

classic_model = DipolePoisson(prior, nside=NSIDE, mask_map=mask_map)
classic_inferer = LikelihoodBasedInferer(x0_flat, classic_model)
classic_inferer.run_ultranest(run_kwargs={'min_ess': 1_000})

# Prepare data for the table
results = [
    [
        "Simulation-based (raw data)",
        (
            f"{raw_sbased_inferer.log_bayesian_evidence:.6g} "
            f"± {raw_sbased_inferer.log_bayesian_evidence_err:.2g}"
        ),
        (
            raw_sbased_inferer.log_bayesian_evidence # type: ignore
          - classic_inferer.log_bayesian_evidence
        ),
        None  # Placeholder for sigma discrepancy
    ],
    [
        "Simulation-based (transformed data)",
        (
            f"{transformed_sbased_inferer.log_bayesian_evidence:.6g} "
            f"± {transformed_sbased_inferer.log_bayesian_evidence_err:.2g}"
        ),
        (
            transformed_sbased_inferer.log_bayesian_evidence # type: ignore
          - classic_inferer.log_bayesian_evidence
        ),
        None
    ],
    [
        "Classic",
        (
            f"{classic_inferer.log_bayesian_evidence:.6g} "
            f"± {classic_inferer.log_bayesian_evidence_err:.2g}"
        ),
        0.0,
        None
    ]
]

headers = ["Method", "log Z ± Error", "Δ log Z", "σ"]
# Format the difference column to 3 decimal places and compute sigma discrepancy
for i, row in enumerate(results):
    row[2] = f"{float(row[2]):.3f}"
    if row[0] == "Classic":
        row[3] = "—"
    else:
        # Calculate sigma discrepancy
        delta = float(row[2])
        err1 = None
        if row[0] == "Simulation-based (raw data)":
            err1 = raw_sbased_inferer.log_bayesian_evidence_err
        elif row[0] == "Simulation-based (transformed data)":
            err1 = transformed_sbased_inferer.log_bayesian_evidence_err
        err2 = classic_inferer.log_bayesian_evidence_err
        assert type(err1) is np.float64; assert type(err2) is np.float64
        if err1 is not None and err2 is not None and (err1**2 + err2**2) > 0:
            sigma = abs(delta) / (err1**2 + err2**2) ** 0.5
            row[3] = f"{sigma:.2f}"
        else:
            row[3] = "N/A"

print(tabulate(results, headers=headers, tablefmt="github"))

raw_sbi_samples = MCSamples(
    samples=raw_samples.detach().cpu().numpy(),
    names=prior.prior_names,
    labels=prior.prior_names
)
transformed_sbi_samples = MCSamples(
    samples=transformed_samples.detach().cpu().numpy(),
    names=prior.prior_names,
    labels=prior.prior_names
)
classic_samples = MCSamples(
    samples=classic_inferer._samples, # type: ignore
    names=prior.prior_names,
    labels=prior.prior_names,
    sampler='nested'
)
g = plots.get_subplot_plotter()
g.triangle_plot(
    [raw_sbi_samples, transformed_sbi_samples, classic_samples],
    filled=True,
    markers=theta0.flatten(),
    marker_args={'lw': 1}, # type: ignore
    legend_labels=['Raw SBI samples', 'Transformed SBI samples', 'Classic samples']
)
plt.show()

