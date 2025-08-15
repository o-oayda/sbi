from typing import Tuple
from healpy import nside2npix
import torch
from torch import Tensor
from nflows.transforms.base import Transform
from torch.utils.data import Dataset, DataLoader, random_split
import math


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

    def forward(self, x, context=None):
        z = 2 * torch.sqrt(x + 0.375)
        logabsdet = self.log_abs_det_jacobian(x, z)
        return z, logabsdet

    def inverse(self, z, context=None):
        x = z**2 / 4 - 0.375
        logabsdet = -self.log_abs_det_jacobian(x, z)
        return x, logabsdet

    def log_abs_det_jacobian(self, x, z):
        return - 0.5 * torch.log(x + 0.375).sum(dim=-1)

class MonotoneAsinh1D(torch.nn.Module):
    """
    y = asinh(s*(x - mu));  s = softplus(w) + s_min  (to keep s>0)
    log|dy/dx| = log s - 0.5*log(1 + (s*(x-mu))^2)
    """
    def __init__(self, s_min=1e-3):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(()))          # scalar
        self.w  = torch.nn.Parameter(torch.zeros(()))          # scalar -> s via softplus
        self.s_min = s_min

    def _s(self):
        return torch.nn.functional.softplus(self.w) + self.s_min

    def forward(self, x):
        # x: arbitrary shape
        s = self._s()
        u = s * (x - self.mu)
        y = torch.asinh(u)
        logdet = torch.log(s).expand_as(x) - 0.5*torch.log1p(u*u)
        return y, logdet

    def inverse(self, y):
        s = self._s()
        u = torch.sinh(y)
        x = self.mu + u / s
        # inverse logdet is negative of forward
        inv_logdet = - (torch.log(s).expand_as(y) - 0.5*torch.log1p((s*(x - self.mu))**2))
        return x, inv_logdet

class LearnableSO4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(0.01 * torch.randn(4,4))

    def forward(self):
        S = self.A - self.A.T # subtracting transpose gives skew-sym. (any nxn)
        return torch.matrix_exp(S) # Q ∈ SO(4)

class HealpixSO4Pyramid(torch.nn.Module):
    def __init__(self, nside_fine: int = 16):
        super().__init__()
        assert nside_fine >= 1
        self.nside_fine = nside_fine
        self.Npix_fine = nside2npix(self.nside_fine)
        print(f'Transform NSIDE: {self.nside_fine}')
        self.dtype = torch.float32

        # nsides per level: [nside_fine, nside_fine/2, ..., 1]
        self.level_nsides = []
        n = nside_fine
        while n >= 1:
            self.level_nsides.append(n)
            n //= 2
        print(f'NSIDE levels: {self.level_nsides}')

        # for each level, we use one learnable SO(4) matrix
        self.SO4_matrices = torch.nn.ModuleList(
            [LearnableSO4() for _ in range(self.levels)]
        )

        # Gaussian stats (for loss only), one (mu, sigma) per level
        self.mu_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(4, dtype=self.dtype))
                for _ in range(self.levels)
            ]
        )
        self.log_sigma_list = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(4, dtype=self.dtype))
                for _ in range(self.levels)
            ]
        )

    @property
    def levels(self):
        return len(self.level_nsides) - 1  # number of downsamplings

    def forward(self, x: torch.Tensor):
        """
        param x: (batches, npix) in nest order
        Returns:
          z: (batches, npix) with layout [a(NSIDE=1) | details (coarse->fine)]
          per_level_quads: list length=levels, each (B, P, 4) AFTER rotation,
            ordered coarse->fine
          logdet: zeros (B,) since by construction the log Jacobian is 0
        """
        batches, npix = x.shape
        assert npix == self.Npix_fine
        v = x
        cur_npix = self.Npix_fine

        quads_f2c = []   # store rotated quads (fine->coarse) to later reverse
        a_list = []

        # LOGIC
        # - for each healpix map, group the 4 pixels corresponding to the children
        #   of the lower res (cur_nside / 2) map (together in nest ordering)

        # - this group goes into 3rd dimension (shape batches, pix / 4, 4)

        # - hit each pixel four vector with matrix vector product, the matrix
        #   being the learnt SO4 matrix; this extracts four coefficients which
        #   live in the 3rd dimension; this is done batchwise with matmul

        # - the four coefficients are appended to quads_f2c, the a coefficients
        #   (first coefficients in 3rd dim) are passed downstream to lower res

        # - at the end, append the coarsest a coefficients + all the 3 detail
        #   coefficients from each level; this ensures the output shape of z
        #   has the same pixel count as the input shape

        # Apply SO(4) to every 4-child group, fine -> coarse
        for lvl in range(self.levels):
            P = cur_npix // 4 # downsampled npix
            quads = v.view(batches, P, 4)                   # (B,P,4)
            # pick the matrix for this fine level_nsides
            # (index from the end so that mixes[0] corresponds to coarsest)
            Q = self.SO4_matrices[self.levels - 1 - lvl]()  # (4,4)
            y = torch.matmul(quads, Q.T)                    # (B,P,4)
            a = y[..., 0]                                   # (B,P)
            quads_f2c.append(y)
            a_list.append(a)
            v = a
            cur_npix = P

        a_coarse = v  # (B, 12)

        # Build z: keep a_coarse, then details from coarse->fine
        # (drop the a's there)
        z_parts = [a_coarse.reshape(batches, -1)]  # list[ (B, 12) ]
        for y in reversed(quads_f2c):              # now coarse->fine
            z_parts.append(y[...,1:].reshape(batches, -1))
        z = torch.cat(z_parts, dim=1)
        logdet = torch.zeros(batches, dtype=x.dtype, device=x.device)

        per_level_quads = list(reversed(quads_f2c))  # coarse->fine order for loss
        return z, per_level_quads, logdet

    def inverse(self, z: torch.Tensor):
        """Exact inverse (uses the same Q_ell); logdet=0."""
        batches, npix = z.shape
        assert npix == self.Npix_fine

        # parents per fine->coarse level
        P_levels = []
        cur = self.Npix_fine
        for _ in range(self.levels):
            P_levels.append(cur // 4)
            cur //= 4

        # parse z
        offset = 12
        a = z[:, :offset]  # (B,12)

        details_c2f = []
        for P in P_levels[::-1]:  # coarse->fine
            dsize = P * 3
            d = z[:, offset:offset+dsize].view(batches, P, 3)
            offset += dsize
            details_c2f.append(d)

        # reconstruct coarse->fine
        v = a
        for lvl, d in enumerate(details_c2f):     # lvl goes 0..coarsest to finer
            Q = self.SO4_matrices[lvl]()                 # this indexing matches coarse->fine
            y = torch.cat([v.unsqueeze(-1), d], dim=-1)   # (B,P,4)
            quads = torch.matmul(y, Q)            # inverse of (· @ Q.T) is (· @ Q)
            v = quads.reshape(batches, -1)
        x_rec = v
        logdet = torch.zeros(batches, dtype=z.dtype, device=z.device)

        return x_rec, logdet

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

def loss_balanced(per_level_quads, mu_list, log_sigma_list):
    """
    Average the mean NLL per level so all scales weigh equally.
    per_level_quads: list length L, each (B, P_l, 4)   (coarse->fine)
    mu_list/log_sigma_list: list length L, each (4,)
    """
    level_losses = []
    for lvl, y in enumerate(per_level_quads):
        mu, log_sigma = mu_list[lvl], log_sigma_list[lvl]
        nll = gaussian_nll(y, mu, log_sigma)    # (B, P_l)
        level_losses.append(nll.mean())              # scalar
    return torch.stack(level_losses).mean()          # scalar

def learn_transformation(
        model: HealpixSO4Pyramid,
        data: Tensor,
        batch_size: int = 128,
        epochs: int = 3,
        lr=0.001,
        validation_frac=0.1,
        min_delta_rel=0.001, 
        patience=100
) -> Tuple[HealpixSO4Pyramid, dict, dict]:
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
        list(model.SO4_matrices.parameters())
      + list(model.mu_list)
      + list(model.log_sigma_list), lr=lr
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
                _, per_level_quads, _ = model(D)   # list of (B, P_l, 4)
                loss = loss_balanced(
                    per_level_quads, 
                    model.mu_list, 
                    model.log_sigma_list
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
            _, per_level_quads, _ = model(D)
            loss = loss_balanced(
                per_level_quads,
                model.mu_list,
                model.log_sigma_list
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
