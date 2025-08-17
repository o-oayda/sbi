from typing import Optional, Tuple, cast, Literal
from healpy import isnsideok, nside2npix
from numpy.typing import NDArray
import torch
from torch import Tensor
from nflows.transforms.base import Transform
from torch.utils.data import Dataset, DataLoader, random_split
import math
import numpy as np
from dipolesbi.tools.utils import softplus_pos
import torch.nn.functional as F


class LogAffineTransform(Transform):
    def __init__(self, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x, context=None) -> Tuple[Tensor, Tensor]: # type: ignore
        z = (torch.log1p(x) - self.mu) / self.sigma
        logabsdet = (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)
        return z, logabsdet

    def inverse(self, z, context=None) -> Tuple[Tensor, Tensor]: # type: ignore
        x = torch.expm1(z * self.sigma + self.mu)
        logabsdet = -(-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)
        return x, logabsdet

    def log_abs_det_jacobian(self, x, z):
        return (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)

class AnscombeTransform(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, x, context=None): # type: ignore
        z = 2 * torch.sqrt(x + 0.375)
        logabsdet = self.log_abs_det_jacobian(x, z)
        return z, logabsdet

    def inverse(self, z, context=None): # type: ignore
        x = z**2 / 4 - 0.375
        logabsdet = -self.log_abs_det_jacobian(x, z)
        return x, logabsdet

    def log_abs_det_jacobian(self, x, z):
        return - 0.5 * torch.log(x + 0.375).sum(dim=-1)

class LearnableSOn(torch.nn.Module):
    def __init__(self, n: int = 4):
        super().__init__()
        self.A = torch.nn.Parameter(0.01 * torch.randn(n,n))

    def forward(self):
        S = self.A - self.A.T # subtracting transpose gives skew-sym. (any nxn)
        return torch.matrix_exp(S) # Q ∈ SO(4)

class LearnableNormaliser(torch.nn.Module):
    def __init__(self, init_eps: float = 0.001) -> None:
        super().__init__()
        self.log_kappa = torch.nn.Parameter(torch.zeros(1))
        self.log_eps = torch.nn.Parameter(torch.log(torch.as_tensor(init_eps)))
        self.beta0 = torch.nn.Parameter(torch.zeros(1))

    def forward(
            self, 
            coarse_coefficients: Tensor, 
            detail_coefficients: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        kappa = torch.exp(self.log_kappa)
        eps = torch.exp(self.log_eps)
        a_pos = F.softplus(coarse_coefficients)
        s = torch.sqrt(eps + kappa * a_pos)
        m = self.beta0 + torch.ones_like(coarse_coefficients) # (n_batches, n_pix)

        a_norm = coarse_coefficients
        d_norm = (detail_coefficients - m.unsqueeze(-1)) / s.unsqueeze(-1) # (n_batches, n_pix, n_coefficients)

        n_detail_coefficients = detail_coefficients.shape[-1]
        logdet = - n_detail_coefficients * torch.sum(torch.log(s + 1e-12), dim=1) # (n_batches,)

        return a_norm, d_norm, logdet

class HealpixSOPyramid(torch.nn.Module):
    def __init__(
            self,
            nside_fine: int = 16,
            nside_at_levels: Optional[list[int]] = None
    ) -> None:
        super().__init__()
        self.nside_fine = nside_fine
        self.npix_fine = nside2npix(self.nside_fine)
        print(f'Input NSIDE: {self.nside_fine}')
        self.dtype = torch.float32

        # compute nside per level; by default nside is reduced by a factor of
        # 2 per level unless downgrade_factors is set
        if nside_at_levels is None:
            self.n_downscales = int(np.log2(nside_fine))
            self.level_nsides = list(
                reversed([2**n for n in range(self.n_downscales+1)])
            )
        else:
            # assert nside_at_levels[-1] == 1, 'Final nside must be 1.'
            assert nside_at_levels[0] == nside_fine, (
                'First nside must be equal to input map nside.'
            )
            nside_ok = cast(NDArray[np.bool_], isnsideok(nside_at_levels))
            assert nside_ok.all(), 'Invalid nside in nside_at_levels.'

            self.level_nsides = nside_at_levels

        self.out_nside = self.level_nsides[-1]
        self.out_npix = nside2npix(self.out_nside)
            
        self.downscale_factors: list[int] = [
            nside2npix(in_nside) // nside2npix(out_nside)
            for in_nside, out_nside in zip(
                self.level_nsides[:-1], self.level_nsides[1:]
            )
        ]
        print(f'NSIDE levels: {self.level_nsides}')
        print(f'Downscale factors: {self.downscale_factors}') # len(level_nsides)-1

        # for each level, we choose a matrix in SO(n) such that n corresponds
        # to the downscale factor from coarse to fine
        self.SO_matrices = torch.nn.ModuleList(
            [LearnableSOn(n) for n in self.downscale_factors]
        )

        # # Gaussian stats for loss function, one (mu, sigma) per level
        self.mu_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(n, dtype=self.dtype))
                for n in self.downscale_factors
            ]
        )
        self.log_sigma_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(n, dtype=self.dtype))
                for n in self.downscale_factors
            ]
        )
        # self.normalisation_couplers = torch.nn.ModuleList(
        #     [
        #         LearnableNormaliser() for _ in self.downscale_factors
        #     ]
        # )
        # self.alpha_list = torch.nn.ParameterList(
        #     [
        #         torch.nn.Parameter(torch.zeros(n, dtype=self.dtype))
        #         for n in self.downscale_factors
        #     ]
        # )
        # self.nu_list = torch.nn.ParameterList(
        #     [
        #         torch.nn.Parameter(30 * torch.rand(n, dtype=self.dtype))
        #         for n in self.downscale_factors
        #     ]
        # )

    @property
    def levels(self):
        return len(self.level_nsides) - 1  # number of downsamplings

    @property
    def parents_at_levels(self):
        P_levels = []
        cur = self.npix_fine
        for dscale_factor in self.downscale_factors:
            P_levels.append(cur // dscale_factor)
            cur //= dscale_factor
        return P_levels

    def forward(self, x: torch.Tensor) -> tuple[Tensor, list, list, Tensor]:
        """
        :param x: (n_batches, n_pix) in nest ordering
        :returns:
          z: (batches, npix) with layout [a(NSIDE=1) | details (coarse->fine)]
          per_level_coefficients: list length=levels, each (B, P, n_coefficients) 
            ordered fine->coarse.
          logdets_at_each_level: value of log determinant Jacobian at each level
          logdet: (n_batches,) value of log determinant Jacobian, 0 by construction
        """
        batches, npix = x.shape
        assert npix == self.npix_fine
        downstream_coefficients = x
        cur_npix = self.npix_fine

        coefficients_fine2coarse = [] 
        a_list = []
        logdets_at_each_level = []

        # LOGIC
        # - for each healpix map, group the pixels corresponding to the children
        #   of the lower res map (potentially many orders lower)

        # - this group goes into 3rd dimension
            # (shape batches, pix / n_coefficients, n_coefficients)

        # - hit each pixel four vector with matrix vector product, the matrix
        #   being the learnt SOn matrix; this extracts n coefficients which
        #   live in the 3rd dimension; this is done batchwise with matmul

        # - the n coefficients are appended to coefficients_fine2coarse,
            # the a coefficients (first coefficients in 3rd dim) are passed
            # downstream to lower res

        # - at the end, append the coarsest a coefficients + all the n-1 detail
        #   coefficients from each level; this ensures the output shape of z
        #   has the same pixel count as the input shape

        # Apply SO(n) to every n-child group, fine -> coarse
        logdet = torch.zeros(batches, dtype=x.dtype, device=x.device)

        for level, factor in zip(range(self.levels), self.downscale_factors):
            P = cur_npix // factor
            child_pixels = downstream_coefficients.view(batches, P, factor)

            Q = self.SO_matrices[level]()

            y = torch.matmul(child_pixels, Q.T)
            a = y[..., 0]
            a_for_cond = y[..., :1]
            d = y[..., 1:]

            y = torch.cat([a_for_cond, d], dim=-1)

            per_level_logdet = 0.
            logdets_at_each_level.append(per_level_logdet)
            logdet += per_level_logdet

            coefficients_fine2coarse.append(y)
            a_list.append(a)
            downstream_coefficients = a
            cur_npix = P

        a_coarse = downstream_coefficients # (batches, out_npix)

        # Build z: keep a_coarse, then details from coarse->fine
        # (drop the a's there)
        z_parts = [a_coarse.reshape(batches, -1)]  # list[ (B, out_npix) ]
        for y in reversed(coefficients_fine2coarse): # now coarse->fine
            z_parts.append(y[...,1:].reshape(batches, -1))
        z = torch.cat(z_parts, dim=1)

        per_level_coefficients = coefficients_fine2coarse  # fine to coarse ordering
        return z, per_level_coefficients, logdets_at_each_level, logdet

    def inverse(self, z: torch.Tensor):
        """
        Exact inverse (uses the same SO(n) matrix); logdet=0.
        """
        batches, npix = z.shape
        assert npix == self.npix_fine

        # parents per fine->coarse level
        P_levels = self.parents_at_levels

        # parse z
        # z = [a_coarse | d_coarse ... d_fine ]
        offset = self.out_npix
        a = z[:, :offset]  # (B, out_npix)

        details_coarse2fine = []
        # iterate from high resolution to low resolution
        for npix, dscale_factor in zip(P_levels[::-1], self.downscale_factors[::-1]):
            dsize = npix * (dscale_factor - 1) # detail coefficients always 1 less 
            d = z[:, offset:offset+dsize].view(batches, npix, dscale_factor - 1)
            offset += dsize
            details_coarse2fine.append(d)

        # reconstruct coarse->fine
        downstream_coefficients = a
        inv_logdet = torch.zeros(batches, dtype=z.dtype, device=z.device)

        # reconstruct coarse to fine
        for lvl, d_out in enumerate(details_coarse2fine):
            # y_out is what forward produced after coupling: [a | d_out]
            a_here = downstream_coefficients.unsqueeze(-1) # (B, npix_level, 1)
            y_pre = torch.cat([a_here, d_out], dim=-1)  # (B, P_l, n_coefficients)

            # undo the SO(n) matrix product
            Q = self.SO_matrices[self.levels - 1 - lvl]() # inverse of (· @ Q.T) is (· @ Q)
            coefficients = torch.matmul(y_pre, Q) # (B, npix_level, n_coefficients)

            # upsample to next finer 'a'
            downstream_coefficients = coefficients.reshape(batches, -1)

        x_rec = downstream_coefficients
        return x_rec, inv_logdet

class MapDataset(Dataset):
    def __init__(self, D_all: torch.Tensor):
        """
        Interface for pytorch.

        D_all: (N, Npix) torch tensor in NEST order.
        """
        self.D = D_all
    def __len__(self): return self.D.shape[0]
    def __getitem__(self, i): return self.D[i]

def gaussian_nll(y, mu, log_sigma):
    # diagonal Gaussian NLL on last dim (size 4)
    inv_var = torch.exp(-2*log_sigma)
    nll = 0.5 * ((y - mu)**2 * inv_var + 2*log_sigma + math.log(2*math.pi))
    return nll.sum(dim=-1)  # sum over 4 dims

def studentst_nll(y, mu, log_sigma, nu):
    sigma = torch.exp(log_sigma)
    r = (y - mu) / sigma
    nll = 0.5 * (nu + 1) * torch.log(1 + r**2 / nu) + torch.log(sigma)
    return nll.sum(dim=-1)

def skew_gaussian_nll(y, mu, log_sigma, alpha):
    sigma = torch.exp(log_sigma)
    sigma = softplus_pos(sigma)
    delta = torch.tanh(alpha)
    alpha = delta / torch.clamp(torch.sqrt(1 - delta*delta), min=1e-8)

    r = (y - mu) / sigma
    nll = 0.5 * r ** 2 + log_sigma - torch.special.log_ndtr(alpha * r)
    return nll.sum(dim=-1)

def compute_nll(
        per_level_quads: Tensor,
        distribution: Literal['gaussian', 'studentst', 'skew_gaussian'] = 'gaussian',
        **distribution_kwargs
) -> Tensor:
    '''
    Compute the mean NLL per level (averaging across these again) given a choice
    of underlying distribution.

    :param per_level_quads: Coefficients computed at each level, of shape
        (n_batches, npix_level, n_coefficients).
    :param distribution: Underlying distribution for loss function.
    :param **distribution_kwargs: Kwargs to pass to distribution.
    '''
    distribution_to_nll_func = {
        'gaussian': gaussian_nll,
        'studentst': studentst_nll,
        'skew_gaussian': skew_gaussian_nll
    }
    valid_distribution_kwargs = {
        'gaussian': ['mu', 'log_sigma'],
        'studentst': ['mu', 'log_sigma', 'nu'],
        'skew_gaussian': ['mu', 'log_sigma', 'alpha']
    }
    kwargs = valid_distribution_kwargs[distribution]
    for kwarg in kwargs:
        assert kwarg in distribution_kwargs, f'Invalid kwargs for {distribution}.'

    loss_function = distribution_to_nll_func[distribution]
    level_losses = []
    for level, y in enumerate(per_level_quads):
        nll = loss_function(
            y,
            **{kwarg: val[level] for kwarg, val in distribution_kwargs.items()}
        )
        level_losses.append(nll.mean())
    return torch.stack(level_losses).mean()

def learn_transformation(
        model,
        data: Tensor,
        batch_size: int = 128,
        epochs: int = 3,
        lr=0.001,
        validation_frac=0.1,
        min_delta_rel=0.001, 
        patience=100
) -> Tuple[HealpixSOPyramid, dict, dict]:
    dataset = MapDataset(data)
    n_simulations = len(dataset)
    n_validation = int(validation_frac * n_simulations)
    n_train = n_simulations - n_validation
    training_set, validation_set = random_split(dataset, [n_train, n_validation]) 

    training_loader = DataLoader(
        training_set,
        batch_size=batch_size, 
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True
    )

    opt = torch.optim.Adam(
        list(model.SO_matrices.parameters())
      + list(model.mu_list)
      + list(model.log_sigma_list),
        lr=lr
    )

    best_val = float("inf")
    best_state = None
    steps_since_best = 0
    history = {"train": [], "val": []}

    def evaluate(loader):
        model.eval()
        tot = 0.0
        cnt = 0
        with torch.no_grad():
            for D in loader:
                _, per_level_quads, *_ = model(D)   # list of (B, P_l, 4)
                loss = compute_nll(
                    per_level_quads, 
                    distribution='gaussian',
                    mu=model.mu_list, 
                    log_sigma=model.log_sigma_list,
                )
                bs = D.size(0)
                tot += float(loss) * bs
                cnt += bs
        return tot / max(1, cnt)

    # Initial validation (optional)
    val0 = evaluate(validation_loader)
    best_val = val0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        print(f'Start of epoch {epoch}.')
        model.train()
        for D in training_loader:
            _, per_level_quads, *_ = model(D)
            loss = compute_nll(
                per_level_quads, 
                distribution='gaussian',
                mu=model.mu_list, 
                log_sigma=model.log_sigma_list,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            history["train"].append(float(loss.detach()))

            # Periodic val check (every ~1 epoch worth of samples / 5, say)
            if len(history["train"]) % max(1, (len(training_loader)//5 or 1)) == 0:
                val = evaluate(validation_loader)
                history["val"].append(val)

                # Early stopping logic (relative improvement)
                rel_impr = (best_val - val) / max(best_val, 1e-12)
                if rel_impr > min_delta_rel:
                    best_val = val
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    steps_since_best = 0
                else:
                    steps_since_best += 1
                    if steps_since_best >= patience:
                        # Restore and exit
                        model.load_state_dict(best_state, strict=True)
                        return (
                            model, 
                            history,
                            {"best_val": best_val, "stopped_early": True}
                        )

    # End of epochs: restore best
    model.load_state_dict(best_state, strict=True)
    return model, history, {"best_val": best_val, "stopped_early": False}
