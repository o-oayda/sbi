from typing import Optional
from torch.types import Tensor
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

# TODO
# - compare with no Affine transform (sbi native structured)
# - simulation spam?

LOAD_PATH_RAW: Optional[str] = None # 'raw_NLE_nside8.pkl'
LOAD_PATH_TRANS: Optional[str] = None # 'trans_NLE_nside8.pkl'

## ---------------------------------------------------
## MODEL PARAMETERS
# nside 4 ok
# nside 16 we start to see very significant discrepancy with log z
# transforming the data seems to make no difference or is slightly worse
NSIDE = 16
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
    n_workers=32,
    simulation_batch_size=100
)

## AFFINE TRANSFORM
# Average over all samples for batch std.
logx = torch.log1p(x)
mu = torch.mean(logx)
sample_std = torch.std(logx, dim=1)
sample_std[sample_std < 1e-14] = 1e-14
t_std = torch.mean(sample_std)
transform = LogAffineTransform(mu, t_std, learn_mu=False) 
z, _ = transform.forward(simulator.x)

# MOCK TRUE SAMPLE
x0 = model.generate_dipole(*theta0)
x0_flat = x0.flatten()
mask_map = ~np.isnan(x0_flat)
x0_truncated = x0_flat[mask_map]
z0_truncated, _ = transform.forward(torch.as_tensor(x0_truncated))
z0_truncated = z0_truncated.numpy()
hp.projview(x0_flat, nest=True)
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
    raw_inferer.run_healpix_sbi(
        estimator_type='NLE', 
        load_simulations_in_vram=False, 
        training_device='cuda',
        nan_handle_method='truncate',
        z_score_x='structured'
    )
    raw_inferer.posterior.to('cpu') # type: ignore
    raw_inferer.plot_loss_curve()
raw_samples = raw_inferer.sample_amortized_posterior(
    torch.as_tensor(x0_truncated),
    n_samps=10_000
)

transformed_inferer = LikelihoodFreeInferer(transformed_simulator)
if LOAD_PATH_TRANS is not None:
    transformed_inferer.load_posterior(LOAD_PATH_TRANS)
else:
    transformed_inferer.run_healpix_sbi(
        estimator_type='NLE', 
        load_simulations_in_vram=False, 
        training_device='cuda',
        nan_handle_method='truncate',
        z_score_x=None
    )
    transformed_inferer.posterior.to('cpu') # type: ignore
    transformed_inferer.plot_loss_curve()
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
    _, logabsdet = transform.inverse(z)
    # -ve sign for log abs det
    return transformed_inferer.posterior.potential(theta, z) - prior.log_prob(theta) - logabsdet

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

