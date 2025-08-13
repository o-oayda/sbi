
from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import torch
import torch.nn as nn
import math
from torch import Tensor
from sbi.neural_nets import likelihood_nn
from sbi.inference import NLE
from corner import corner

from nflows.transforms import CompositeTransform, AffineCouplingTransform, RandomPermutation
from nflows.distributions.normal import StandardNormal
from nflows.flows import Flow
from nflows.nn import nets
from nflows.transforms.base import Transform


class ContextIgnoreWrapper(nn.Module):
    """Wraps a net to ignore context but still register parameters."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, context=None):
        return self.net(x)

def make_realnvp_flow(num_features: int,
                      mu = Tensor,
                      sigma = Tensor,
                      num_coupling_layers: int = 8,
                      hidden_features: int = 256,
                      device: str = "cpu"):
    """
    Build a RealNVP-style flow using AffineCouplingTransform and RandomPermutation.
    Uses nflows.nn.nets.MLP as the transform net factory.
    """
    transforms = []
    # binary alternating mask 0/1 0/1...
    mask = (torch.arange(num_features) % 2).to(torch.int64)

    norm_transform = LogAffineTransform(mu, sigma)

    for i in range(num_coupling_layers):
        # create a factory function for the transform net
        def create_net(in_features, out_features):
            # nflows' nets.MLP expects shapes as tuples for in_shape/out_shape
            net = nets.MLP(
                in_shape=(in_features,),
                out_shape=(out_features,),
                hidden_sizes=[hidden_features, hidden_features],
                activation=nn.ReLU()
            )
            return ContextIgnoreWrapper(net)

        transforms.append(
            AffineCouplingTransform(mask=mask, transform_net_create_fn=create_net)
        )
        # flip mask for next layer (so other dimensions are transformed)
        mask = 1 - mask
        # add a permutation to mix components
        transforms.append(RandomPermutation(features=num_features))

    all_transforms = [norm_transform] + transforms
    transform = CompositeTransform(all_transforms)
    base_dist = StandardNormal(shape=[num_features])
    flow = Flow(transform, base_dist).to(device)
    return flow

class LogAffineTransform(Transform):
    def __init__(self, mu_init: Tensor, sigma: Tensor, learn_mu=True):
        super().__init__()
        if learn_mu:
            # Learnable mean parameter
            self.mu = nn.Parameter(mu_init.clone().detach())
        else:
            # Fixed mean (non-learnable)
            self.register_buffer("mu", mu_init.clone().detach())

        # σ stays fixed for stability
        self.register_buffer("sigma", sigma.clone().detach())

    def forward(self, x, context=None):
        z = (torch.log1p(x) - self.mu) / self.sigma
        logabsdet = (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)
        return z, logabsdet

    def inverse(self, z, context=None):
        x = torch.expm1(z * self.sigma + self.mu)
        logabsdet = -(-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)
        return x, logabsdet

    def log_abs_det_jacobian(self, x, z):
        return (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)

# class LogAffineTransform(Transform):
#     def __init__(self, mu, sigma):
#         super().__init__()
#         self.mu = 0
#         self.sigma = sigma
#
#     def forward(self, x, context=None):
#         # x assumed positive (counts)
#         z = (torch.log1p(x) - self.mu) / self.sigma
#         logabsdet = self.log_abs_det_jacobian(x, z) 
#         return z, logabsdet
#
#     def inverse(self, z, context=None):
#         x = torch.expm1(z * self.sigma + self.mu)
#         # The inverse log det jacobian is negative of forward's log det jacobian evaluated at inverse
#         logabsdet = -self.log_abs_det_jacobian(x, z)
#         return x, logabsdet
#
#     def log_abs_det_jacobian(self, x, z):
#         # For each dimension:
#         # dz/dx = 1 / [(1+x) * sigma]
#         # log|dz/dx| = -log(1+x) - log(sigma)
#         return (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)

# class AnscombeTransform(Transform):
#     def __init__(self, mu, sigma):
#         super().__init__()
#         self.mu = mu
#         self.sigma = sigma
#
#     def forward(self, x, context=None):
#         z = 2 * torch.sqrt(x + 0.375)
#         logabsdet = self.log_abs_det_jacobian(x, z)
#         return z, logabsdet
#
#     def inverse(self, z, context=None):
#         x = z**2 / 4 - 0.375
#         logabsdet = self.log_abs_det_jacobian(x, z)
#         return x, logabsdet
#
#     def log_abs_det_jacobian(self, x, z):
#         return - 0.5 * torch.log(x + 0.375).sum(dim=-1)

class AnscombeTransform(Transform):
    def __init__(self, learnable_affine=True, eps=1e-6):
        super().__init__()
        self.eps = eps  # numerical stability

        if learnable_affine:
            self.shift = nn.Parameter(torch.zeros(1))
            self.log_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("shift", torch.zeros(1))
            self.register_buffer("log_scale", torch.zeros(1))

    def forward(self, x, context=None):
        """
        x: raw counts, shape (batch, features), assumed >= 0.
        """
        # 1. Variance-stabilizing transform
        z = 2.0 * torch.sqrt(x + 3.0 / 8.0 + self.eps)

        # 2. Learnable affine
        z = (z + self.shift) * torch.exp(self.log_scale)

        # 3. Log|det J|: derivative of Anscombe w.r.t x, then affine
        # d/dx Anscombe = 1 / sqrt(x + 3/8)
        log_jac_anscombe = -0.5 * torch.log(x + 3.0 / 8.0 + self.eps)

        # affine derivative scaling
        log_jac_affine = self.log_scale.expand_as(z)

        logabsdet = (log_jac_anscombe + log_jac_affine).sum(dim=-1)

        return z, logabsdet

    def inverse(self, z, context=None):
        # Undo affine
        z = z * torch.exp(-self.log_scale) - self.shift

        # Approximate inverse Anscombe:
        # Here we use the simple reverse: ((z/2)^2 - 3/8)
        # For high precision, one could use iterative inversion.
        x = (0.25 * z ** 2) - 3.0 / 8.0
        x = torch.clamp(x, min=0.0)

        return x

    def log_abs_det_jacobian(self, x, z):
        # This will only be called in some contexts — we match forward's logic
        log_jac_anscombe = -0.5 * torch.log(x + 3.0 / 8.0 + self.eps)
        log_jac_affine = self.log_scale.expand_as(z)
        return (log_jac_anscombe + log_jac_affine).sum(dim=-1)

# -----------------------
# Training loop for the unconditional flow
# -----------------------
def train_flow_on_simulations(
    flow: Flow,
    data: Tensor,
    nsims: int = 2000,
    batch_size: int = 64,
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu"
): 
    """
    Trains the unconditional flow on nsims draws from prior and simulator.
    prior_sampler: a function that returns theta numpy arrays shape (ntheta,)
    simulator_fn: maps theta -> D (numpy array shape (npix,))
    """
    D_tensor = data
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-6)
    n_batches = math.ceil(nsims / batch_size)

    n_ll = []
    for epoch in range(n_epochs):
        perm = torch.randperm(nsims)
        running_loss = 0.0
        for i in range(0, nsims, batch_size):
            idx = perm[i:i+batch_size]
            batch = D_tensor[idx]
            # maximize log_prob -> minimize negative
            loss = -flow.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5)
            optimizer.step()
            running_loss += float(loss.item()) * batch.shape[0]
        avg_nll = running_loss / nsims
        print(f"[Flow] Epoch {epoch+1}/{n_epochs}  NLL={avg_nll:.4f}")
        n_ll.append(avg_nll)
    return flow, n_ll 

def D_to_z_and_logabsdet(flow: Flow, D: Tensor, device: str = "cpu"):
    """
    Given a single map D (numpy shape (npix,)), returns:
      z (numpy shape (npix,)), logabsdet (float)
    Works on single example; vectorized versions easy to add.
    """
    flow = flow.to(device)
    x = D  # [1, npix]
    # Use the transform internals to get z and logabsdet
    with torch.no_grad():
        z, logabs = flow._transform.forward(x)
    return z, logabs

def z_to_D(flow: Flow, z: Tensor, device: str = "cpu"):
    """
    Invert transform: z (np) -> D (np)
    """
    flow = flow.to(device)
    z_t = z
    with torch.no_grad():
        x, logabs = flow._transform.inverse(z_t)
    return x, logabs

if __name__ == '__main__':
# flow = make_realnvp_flow(mu=mu, sigma=t_std, num_features=NPIX, num_coupling_layers=8, hidden_features=256)
# flow, n_ll = train_flow_on_simulations(
#     flow=flow,
#     data=x,
#     nsims=50_000,
#     batch_size=128,
#     n_epochs=75,
#     lr=1e-3
# )
# plt.plot(n_ll)
# plt.show()
# x_test = torch.as_tensor(x0.reshape(1, NPIX))
# z_test , _ = D_to_z_and_logabsdet(flow, x_test)
# x_test_rec, _ = z_to_D(flow, z_test)

# hp.projview(x_test.squeeze(0).numpy())
# hp.projview(z_test.squeeze(0).numpy())
# hp.projview(x_test_rec.squeeze(0).numpy())
# plt.show()
## ---------------------------------------------------
## MODEL PARAMETERS
# nside 4 ok
    NSIDE = 32
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

    model = SimpleDipoleMap(nside=NSIDE)
    if EQUATOR_MASK != 0.:
        model.equatorial_plane_mask(EQUATOR_MASK)
    simulator = Simulator(prior, model.generate_dipole)
    theta, x = simulator.make_batch_simulations(
        n_simulations=50_000, 
        n_workers=32,
        simulation_batch_size=100
    )

    x0 = model.generate_dipole(*theta0)
    x0_flat = x0.flatten()
    mask_map = ~np.isnan(x0_flat)
    x0_truncated = x0_flat[mask_map]
    hp.projview(x0_flat, nest=True)
    plt.show()
# structured mean
    logx = torch.log1p(x)
    mu = torch.mean(logx)
    sample_std = torch.std(logx, dim=1)
    sample_std[sample_std < 1e-14] = 1e-14
# Average over all samples for batch std.
    t_std = torch.mean(sample_std)
    transform = LogAffineTransform(mu, t_std, learn_mu=False) 
    z, _ = transform.forward(x)
    prior.to('cpu')
    sim = Simulator(prior, model.generate_dipole)
# z, _ = D_to_z_and_logabsdet(flow, x)
    sim.sbi_processed_prior.to('cuda')
    neural_posterior = likelihood_nn(
        model='maf',
        z_score_x=None,
        z_score_theta='independent'
    )
    inference = NLE(
        prior=sim.sbi_processed_prior,
        density_estimator=neural_posterior,
        device='cuda'
    )

    inference = inference.append_simulations(
        x=z,
        theta=theta,
        data_device='cpu'
    )
    density_estimator = inference.train(
        show_train_summary=True,
        stop_after_epochs=30
    )
    posterior = inference.build_posterior(
        density_estimator=density_estimator, # type: ignore
        prior=sim.sbi_processed_prior,
        sample_with='mcmc',
        mcmc_method='slice_np_vectorized'
    )
    posterior.to('cpu')
    x_test = torch.as_tensor(x0.reshape(1, NPIX))
    z_test , _ = transform.forward(x_test)
    samples = posterior.sample((10_000,), x=z_test)
    corner(samples.numpy(), truths=theta0.flatten())
    plt.show()
