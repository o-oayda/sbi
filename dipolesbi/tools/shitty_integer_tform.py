
# Minimal HEALPix Integer Multiscale Flow (PyTorch)
# -------------------------------------------------
# - Multiscale integer bijection with log-det = 0 (unimodular transforms + integer coupling)
# - Priors: Negative Binomial for coarse; Discrete Laplace for details
# - HEALPix NEST grouping: quads are contiguous (children indices 4p..4p+3)
#
# Usage:
#   flow = HealpixIntegerMultiscaleFlow(L=6, hidden=64, tau=32.0)
#   logp = flow.log_prob(x_int)       # x_int: (B, 12*4^L) Long/Float integers
#   z = flow.forward_transform(x_int) # returns (a, details) for inspection
#   x_rec = flow.inverse_transform(z) # exact inverse
#
# Later extensions:
#   - Switch to OPTION_B transform, or add unimodular "intmix" shears (det=±1)
#   - Insert modulo m after every integer op for true ℤ_m flows
#   - Or wrap inputs with base-B digits to bound alphabets

import math
from typing import List, Tuple
from healpy import nside2npix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.transforms import MapDataset

# ----------------------------
# Small utilities
# ----------------------------

def straight_through_round(x: torch.Tensor) -> torch.Tensor:
    # Straight-through estimator for integer rounding
    return (x - x.detach()) + torch.round(x.detach())

def nest_group_quads(x: torch.Tensor, level: int) -> torch.Tensor:
    """
    Group children as 4-tuples for HEALPix NEST at 'level'.
    x: (B, 12*4^level)
    returns quads: (B, num_parents, 4) with children contiguous.
    """
    B, P = x.shape
    num_parents = (12 * (4 ** (level - 1))) if level > 0 else 12
    assert P == 12 * (4 ** level), f"Shape mismatch: got {P}, expected {12*(4**level)}"
    return x.view(B, num_parents, 4)

def nest_ungroup_quads(quads: torch.Tensor, level: int) -> torch.Tensor:
    """
    Flatten back to (B, 12*4^level) from (B, num_parents, 4).
    """
    B, num_parents, four = quads.shape
    assert four == 4
    return quads.view(B, num_parents * 4)

# ----------------------------
# Unimodular 4x4 transform (OPTION A)
#   a  = x0 + x1 + x2 + x3
#   d1 = x0
#   d2 = x1
#   d3 = x2
# det = -1  (⇒ log|det| = 0)
# ----------------------------

def lifting4_forward_A(quads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # quads: (B, Np, 4)
    x0 = quads[..., 0]
    x1 = quads[..., 1]
    x2 = quads[..., 2]
    x3 = quads[..., 3]
    a  = x0 + x1 + x2 + x3
    d1 = x0
    d2 = x1
    d3 = x2
    return a, d1, d2, d3

def lifting4_inverse_A(a, d1, d2, d3) -> torch.Tensor:
    # returns quads: (B, Np, 4)
    x0 = d1
    x1 = d2
    x2 = d3
    x3 = a - d1 - d2 - d3
    return torch.stack([x0, x1, x2, x3], dim=-1)

# ----------------------------
# Tiny MLP for translation t(A)->R^{3}
# ----------------------------

class TNet(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 3)   # predict integer translations for (d1,d2,d3)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (B, Np) -> output: (B, Np, 3)
        h = F.gelu(self.fc1(a.unsqueeze(-1)))
        h = F.gelu(self.fc2(h))
        t = self.fc3(h)                   # real-valued; will be bounded, then rounded
        return t

# ----------------------------
# Discrete priors
# ----------------------------

def log_nb_pmf(x, r, p, eps=1e-12):
    # x >= 0 integer; r>0, 0<p<1
    x = torch.clamp(x, min=0).to(torch.float32)
    r = torch.clamp(r, min=eps)
    p = torch.clamp(p, min=eps, max=1.0 - eps)
    return (torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1)
            + r * torch.log(p) + x * torch.log(1.0 - p))

def log_discrete_laplace(x, log_b, eps=1e-12):
    # P(k) ∝ alpha^{|k|}, where alpha = exp(-1/b), Z = (1-alpha)/(1+alpha)
    # x ∈ Z (can be negative). Properly normalized.
    b = torch.exp(log_b).clamp_min(eps)
    alpha = torch.exp(-1.0 / b).clamp(max=1.0 - 1e-6)     # avoid alpha→1
    # log Z = log(1-alpha) - log(1+alpha)
    logZ = torch.log1p(-alpha) - torch.log1p(alpha)
    return -torch.abs(x).to(torch.float32) * (1.0 / b) + logZ

# ----------------------------
# Main class
# ----------------------------

class HealpixIntegerMultiscaleFlow(nn.Module):
    """
    Minimal HEALPix-aware integer multiscale flow (bijective; log-det = 0).
    - L levels (N_side = 2^L)
    - Children are grouped as NEST quads.
    - Integer coupling on details with capped translations and straight-through rounding.
    - Priors: NB for final coarse (12 values), Discrete Laplace for details at each level.
    Assumes counts < 256 by default (no modulo applied), but easy to extend.
    """
    def __init__(self, L: int, hidden: int = 64, tau: float = 32.0, use_recenter: bool = True):
        super().__init__()
        assert L >= 1, "Need at least one level"
        self.L = L
        self.P = 12 * (4 ** L)
        self.tau = nn.Parameter(torch.tensor(float(tau)))   # can be made per-level if desired
        self.use_recenter = use_recenter

        # Integer recentering offset (learned or set externally). Start at 0.
        self.offset_int = nn.Parameter(torch.zeros(1, 1))   # scalar offset (integer during use)

        # One T-net per level (condition on A -> translate D=(d1,d2,d3))
        self.tnets = nn.ModuleList([TNet(hidden) for _ in range(L)])

        # Priors:
        # - Final coarse (12 integers): NB params r>0, p in (0,1)
        self.nb_logr   = nn.Parameter(torch.zeros(1, 12))   # r = exp(nb_logr)
        self.nb_logitp = nn.Parameter(torch.zeros(1, 12))   # p = sigmoid(nb_logitp)

        # - Details: per-level Discrete Laplace scale
        self.logb_details = nn.Parameter(torch.zeros(L))    # one b per level (shared across d1,d2,d3)

    # ------------- helpers -------------
    def _cap_and_round_t(self, t_real: torch.Tensor) -> torch.Tensor:
        # t_real: (B, Np, 3)
        t_cap = self.tau * torch.tanh(t_real)
        return straight_through_round(t_cap)  # integer update with STE

    def _apply_integer_coupling_forward(self, A, D_stack, level_idx: int):
        """
        Forward translate: D' = D + round(tau * tanh(TNet(A)))
        A: (B, Np), D_stack: (B, Np, 3)
        """
        t_real = self.tnets[level_idx](A)     # (B, Np, 3)
        t_int  = self._cap_and_round_t(t_real)
        return D_stack + t_int

    def _apply_integer_coupling_inverse(self, A, Dp_stack, level_idx: int):
        t_real = self.tnets[level_idx](A)
        t_int  = self._cap_and_round_t(t_real)
        return Dp_stack - t_int

    # ------------- forward/inverse multiscale -------------
    @torch.no_grad()
    def inverse_transform(self, z: Tuple[torch.Tensor, List[Tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]) -> torch.Tensor:
        """
        z = (a, details) with:
          a: (B, 12) final coarse
          details: list of length L, each is (D1,D2,D3) with shapes (B, Np_level)
        Returns x: (B, 12*4^L) integers (as float tensor when returned).
        """
        a, details = z
        # Enforce float for compute; values are integers by construction
        a = a.clone().to(torch.float32)

        # Synthesis from coarse to fine
        for level in range(1, self.L + 1):
            # Retrieve details for this level (reverse order of forward)
            D1, D2, D3 = details[self.L - level]
            D1 = D1.to(torch.float32); D2 = D2.to(torch.float32); D3 = D3.to(torch.float32)

            # Undo coupling on details
            D_stack = torch.stack([D1, D2, D3], dim=-1)  # (B, Np, 3)
            D_stack = self._apply_integer_coupling_inverse(a, D_stack, level_idx=level-1)
            D1, D2, D3 = D_stack[..., 0], D_stack[..., 1], D_stack[..., 2]

            # Unimodular inverse (OPTION A)
            quads = lifting4_inverse_A(a, D1, D2, D3)      # (B, Np, 4)
            # Ungroup to new fine map
            a = nest_ungroup_quads(quads, level)

        x = a
        if self.use_recenter:
            x = x + torch.round(self.offset_int)  # integer inverse recenter
        return x

    def forward_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor,torch.Tensor,torch.Tensor]]]:
        """
        Analysis transform: x -> (a, details)
        x: (B, 12*4^L), integer tensor (float OK; treated as integers)
        Returns:
          a: (B, 12) final coarse (integers as float)
          details: list length L with (D1,D2,D3), each (B, Np_level)
        """
        # Integer recentering
        if self.use_recenter:
            x = x - torch.round(self.offset_int)  # keep integers

        a = x.to(torch.float32)
        details: List[Tuple[torch.Tensor,torch.Tensor,torch.Tensor]] = []

        # Go from fine (level L) to coarse (1)
        for level in range(self.L, 0, -1):
            quads = nest_group_quads(a, level)           # (B, Np, 4)
            A, D1, D2, D3 = lifting4_forward_A(quads)    # all (B, Np)

            # Integer coupling on details conditioned on A
            D_stack = torch.stack([D1, D2, D3], dim=-1)  # (B, Np, 3)
            D_stack = self._apply_integer_coupling_forward(A, D_stack, level_idx=level-1)
            D1, D2, D3 = D_stack[..., 0], D_stack[..., 1], D_stack[..., 2]

            details.append((D1, D2, D3))
            a = A  # pass coarse up

        # a now has shape (B, 12)
        return a, details

    # ------------- log-prob / sampling -------------
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact discrete log-likelihood under:
          - bijective transform (log-det = 0),
          - NB prior on final coarse,
          - Discrete Laplace priors on details per level.
        Returns (B,) log-prob.
        """
        a, details = self.forward_transform(x)

        # Coarse prior (12 values)
        r = torch.exp(self.nb_logr)                    # (1,12)
        p = torch.sigmoid(self.nb_logitp)              # (1,12)
        # Clamp coarse to nonnegative (counts); if you anticipate negatives, model sign+|a|
        a_clamped = torch.clamp(a, min=0.0)
        logp_coarse = log_nb_pmf(a_clamped, r, p)  # broadcast to (B,12)
        logp = logp_coarse.sum(dim=1)

        # Details priors per level
        for lvl, (D1, D2, D3) in enumerate(details, start=1):
            b_lvl = self.logb_details[lvl-1]
            logp += log_discrete_laplace(D1, b_lvl).sum(dim=1)
            logp += log_discrete_laplace(D2, b_lvl).sum(dim=1)
            logp += log_discrete_laplace(D3, b_lvl).sum(dim=1)

        return logp

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample x by:
          1) Sampling final coarse from NB,
          2) Sampling details per level from Discrete Laplace,
          3) Inverting the transform.
        """
        device = self.nb_logr.device
        # Sample coarse (B,12)
        r = torch.exp(self.nb_logr).to(device)
        p = torch.sigmoid(self.nb_logitp).to(device)
        rate = p / (1.0 - p)
        # Poisson-Gamma mixture sampler for NB:
        lam = torch.distributions.Gamma(concentration=r, rate=rate).sample((batch_size,))  # (B,12)
        a = torch.poisson(lam).to(torch.float32)

        # Sample details list (length L)
        details: List[Tuple[torch.Tensor,torch.Tensor,torch.Tensor]] = []
        for lvl in range(1, self.L + 1):
            b = torch.exp(self.logb_details[lvl-1]).item()
            # Discrete Laplace via two-sided geometric with parameter alpha=exp(-1/b)
            alpha = math.exp(-1.0 / max(b, 1e-6))
            # Geometric sampler on |k| with P(|k|=n) ∝ alpha^n:
            def sample_abs(shape):
                # Inverse CDF for geometric with support {0,1,2,...}: P(N>=n)=alpha^n
                u = torch.rand(*shape, device=device).clamp_(min=1e-6)
                # floor(log(u)/log(alpha)) but alpha in (0,1) ⇒ log(alpha)<0
                return torch.floor(torch.log(u) / math.log(alpha))
            def sample_signed(shape):
                n = sample_abs(shape)
                s = torch.where(torch.rand(*shape, device=device) < 0.5, -1.0, 1.0)
                return (s * n).to(torch.float32)

            # number of parents at this level:
            Np = 12 * (4 ** (self.L - lvl))
            D1 = sample_signed((batch_size, Np))
            D2 = sample_signed((batch_size, Np))
            D3 = sample_signed((batch_size, Np))
            details.append((D1, D2, D3))

        # Inverse transform to x
        z = (a, details)
        x = self.inverse_transform(z)
        return x

if __name__ == '__main__':
    MEAN_DENSITY = 20
    NSIDE = 32
    D = nside2npix(NSIDE)

    mean_count_range = [0.95*MEAN_DENSITY, 1.05*MEAN_DENSITY]
    prior = DipolePrior(mean_count_range=mean_count_range)
    prior.change_kwarg('N', 'mean_density')

    dipole = SimpleDipoleMap(nside=NSIDE)
    simulator = Simulator(prior, dipole.generate_dipole)
    theta, x = simulator.make_batch_simulations(
        n_simulations=1000,
        n_workers=32,
        simulation_batch_size=100
    )

    x_data = MapDataset(x[:800])
    x_validation = MapDataset(x[800:])
    train_loader = DataLoader(x_data, batch_size=20, shuffle=True)
    validation_loader = DataLoader(x_validation, batch_size=20, shuffle=False)

    L = int(math.log2(NSIDE))
    model = HealpixIntegerMultiscaleFlow(L)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100
    nll = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch in train_loader:
            optimizer.zero_grad()
            # log_prob returns (B,) log-likelihoods
            logp = model.log_prob(x_batch)
            loss = -logp.mean()              # negative log-likelihood
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
        avg_loss = total_loss / len(x_data)
        nll.append(avg_loss)
        print(f"Epoch {epoch+1}, NLL = {avg_loss:.3f}")


    # logp = model.log_prob()

#   flow = HealpixIntegerMultiscaleFlow(L=6, hidden=64, tau=32.0)
#   logp = flow.log_prob(x_int)       # x_int: (B, 12*4^L) Long/Float integers
#   z = flow.forward_transform(x_int) # returns (a, details) for inspection
#   x_rec = flow.inverse_transform(z) # exact inverse
# ----------------------------
# Notes on extending beyond counts < 256:
# ----------------------------
# 1) True modulo arithmetic:
#    - After each integer op, do: y = (y % m + m) % m
#    - Require any extra mixing matrix to be invertible in Z_m (for m=2^b, determinant must be odd).
# 2) Base-B digits:
#    - Replace inputs with K base-B digits (bijective), run this same flow on bounded digits as channels,
#      then reconstruct at the end; no change to outer logic.
# 3) Handling negative coarse values:
#    - If upstream recentering or transforms produce negatives in 'a', model sign bit + |a| with NB,
#      or switch to a two-sided count prior (e.g., discrete Laplace) for 'a' as well.
