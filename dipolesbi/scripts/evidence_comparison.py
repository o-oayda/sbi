import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import healpy as hp
from dipolesbi.tools.dataloader import split_train_val
from dipolesbi.tools.healpix_helpers import build_funnel_steps
from dipolesbi.tools.inference import JaxNestedSampler, LikelihoodBasedInferer
from dipolesbi.tools.maps import SimpleDipoleMap
from surjectors.util import named_dataset
from dipolesbi.tools.models import DipolePoisson
from dipolesbi.tools.neural_flows import MAFNeuralLikelihood, MAFSurjectiveNeuralLikelihood
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.transforms import HaarWaveletTransform, HealpixSOPyramid, learn_transformation
from dipolesbi.tools.utils import jax_cart2sph, jax_sph2cart
import torch

rng_key = jax.random.PRNGKey(42)
n = 50_000
torch_training_device = 'cuda'

# healpix Poisson data
NSIDE = 16
ndim=12 * NSIDE**2
TOTAL_SOURCES = 1_920_000
MEAN_DENSITY = TOTAL_SOURCES / hp.nside2npix(NSIDE)
OBSERVER_SPEED = 2.
DIPOLE_LONGITUDE = 215.
DIPOLE_LATITUDE = 40.
N_EVIDENCE = 30
mask_map = np.ones(ndim, dtype=np.bool_)
theta0 = jnp.repeat(
    jnp.asarray(
        [[MEAN_DENSITY, OBSERVER_SPEED, DIPOLE_LONGITUDE, DIPOLE_LATITUDE]]
    ),
    repeats=n,
    axis=0
)

dipole = SimpleDipoleMap(NSIDE)    
mean_count_range = [0.99*MEAN_DENSITY, 1.01*MEAN_DENSITY]
prior = DipolePrior(mean_count_range=mean_count_range)
jax_prior = DipolePriorJax(mean_count_range=mean_count_range)
prior.change_kwarg('N', 'mean_density')
simulator = Simulator(prior, dipole.generate_dipole)
prior_samples, y = simulator.make_batch_simulations(
    n_simulations=n,
    n_workers=32,
    simulation_batch_size=1000
)
prior_samples = jnp.asarray(prior_samples)
y = jnp.asarray(y)
(y_tr, y_val), (t_tr, t_val) = split_train_val(y, prior_samples)

y_mean = y_tr.mean(axis=0); y_std = y_tr.std(axis=0) + 1e-8
# y_mean = y_tr.mean(); y_std = y_tr.std(axis=1).mean()

# shitty at nside=1
last_nside = 4
last_npix = 12 * last_nside**2
tform = HaarWaveletTransform()
def normalise_y(y):
    z, logdet = tform.cycle_healpix_tree(y, last_nside=last_nside)
    return z
unnormalise_y = lambda input: input

detail_sizes = []
ns = NSIDE
while ns > last_nside:
    ns //= 2
    npi = 12 * ns**2
    detail_sizes.append(3*npi)
block_lengths = [last_npix] + detail_sizes # a coeffs + 3 details per level
bounds = np.cumsum(block_lengths)
starts  = np.concatenate(([0], bounds[:-1]))
blocks = [(int(s), int(e)) for (s, e) in zip(starts[1:], bounds[1:])]
assert 12 * NSIDE**2 == sum(block_lengths)
steps = build_funnel_steps(n_coarse=12*last_nside**2, detail_lengths=block_lengths[1:])

# log_det_jac = lambda
# normalise_y = lambda input: (input - y_mean) / y_std
# unnormalise_y = lambda input: y_std * input + y_mean
# log_det_jac = lambda y: - jnp.log(y_std) * jnp.ones(y.shape[-1])

mean_nbar = t_tr[:, 0].mean()
std_nbar = t_tr[:, 0].std()
mean_v = t_tr[:, 1].mean()
std_v = t_tr[:, 1].std()
t_mean = jnp.asarray([mean_nbar, mean_v, 0, 0])
t_std = jnp.asarray([std_nbar, std_v, 1, 1])

def transform_t(t):
    lon = jnp.deg2rad(t[..., 2])
    colat = jnp.pi / 2 - jnp.deg2rad(t[..., 3])
    x, y, z = jax_sph2cart(lon, colat)
    
    t_norm = (t - t_mean) / t_std
    t_transformed = jnp.stack(
        [
            t_norm[:, 0],
            t_norm[:, 1],
            x, y, z
        ],
        axis=1
    )
    return t_transformed

def transform_theta_jax(
        params: dict[str, jnp.ndarray]
) -> jnp.ndarray:
    lon = jnp.deg2rad(params['phi'])
    colat = jnp.pi / 2 - jnp.deg2rad(params['theta'])
    x, y, z = jax_sph2cart(lon, colat)

    N_norm = (params['N'] - t_mean[0]) / t_std[0]
    D_norm = (params['D'] - t_mean[1]) / t_std[1]

    t_transformed = jnp.stack(
        [
            N_norm,
            D_norm,
            x, y, z
        ],
    )
    return t_transformed

def untransform_t(t):
    x, y, z = t[:, 2], t[:, 3], t[:, 4]
    phi, theta = jax_cart2sph(x, y, z)
    lon = jnp.rad2deg(phi)
    lat = 90. - jnp.rad2deg(theta)

    t = t * t_std + t_mean
    t_untransformed = jnp.stack(
        [
            t[:, 0],
            t[:, 1],
            lon,
            lat
        ],
        axis=1
    )
    return t_untransformed

train_data = named_dataset(normalise_y(y_tr), transform_t(t_tr))
val_data = named_dataset(normalise_y(y_val), transform_t(t_val))

# nle = MAFNeuralLikelihood(ndim, n_layers=5)

# nuking the nle with large n_layers and reduction_factor close to 1
# reduces evidence tension
# not sure, the accuracy of the evidence is quite noisy at nside=8
nle = MAFSurjectiveNeuralLikelihood(
    ndim, 
    # n_layers=8, 
    # # data_reduction_factor=0.6, # lower implies larger reduction
    decoder_distribution='gaussian',
    conditioner='made',
    blocks=steps,
    maf_stack_size=15,
    conditioner_n_neurons=128,
    decoder_n_layers=3
)
# low learning rate good, possibly training idiosyncrasies
nle.train(hk.PRNGSequence(2), train_data, val_data, learning_rate=1e-5)
nle.plot_loss_curve()

# OOM with students_t
samples = nle.sample_likelihood_func(random.PRNGKey(2), theta0=transform_t(theta0))
true_mean_likelihood = dipole.generate_dipole(
    *np.asarray(theta0[0, :])[:, None],
    make_poisson_draws=False
).squeeze()
mean_samples = unnormalise_y(samples).mean(axis=0)

hp.projview(
    mean_samples, nest=True, graticule=True, sub=211,
    title='NLE mean P(D | theta)'
)
hp.projview(
    true_mean_likelihood, nest=True, graticule=True, sub=212, 
    title='True mean P(D | theta)'
)
plt.show()

evidence_diff = []
sigmas = []
for _ in range(N_EVIDENCE):
    x0 = dipole.generate_dipole(
        *np.asarray(
            theta0[0, :]
        ).reshape(4, 1)
    )
    data = normalise_y(x0)
    _, logdetjac = tform.cycle_healpix_tree(jnp.asarray(x0), last_nside=last_nside)
    def lnlike_jax(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        theta: jnp.ndarray = transform_theta_jax(params)

        log_like = nle.evaluate_lnlike(theta[None, :], data)
        # total_log_like = (log_like + log_det_jac(data).sum(axis=-1)).squeeze()
        total_log_like = (log_like + logdetjac).squeeze()

        return total_log_like

    jax_ns = JaxNestedSampler(lnlike_jax, jax_prior)
    jax_ns.setup(rng_key, n_live=500, n_delete=50)
    nested_samples = jax_ns.run()

    classic_model = DipolePoisson(prior, nside=NSIDE, mask_map=mask_map)
    classic_inferer = LikelihoodBasedInferer(x0.squeeze(), classic_model)
    classic_inferer.run_ultranest()

    print(
        f"NLE Log Evidence: {nested_samples.logZ():.2f} "
        f"± {nested_samples.logZ(100).std():.2f}" # type: ignore
    )
    print(
        f"Classic Log Evidence: {classic_inferer.log_bayesian_evidence:.2f} "
        f"± {classic_inferer.log_bayesian_evidence_err:.2f}" # type: ignore
    )
    diff = nested_samples.logZ() - classic_inferer.log_bayesian_evidence # type: ignore
    sigma = (
        np.abs(diff) # type: ignore
      / np.sqrt(nested_samples.logZ(1000).std()**2 + classic_inferer.log_bayesian_evidence_err**2)) # type: ignore
    print(f'Tension: {sigma}')
    evidence_diff.append(diff)
    sigmas.append(sigma)
