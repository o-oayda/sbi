
# Minimal working example: HEALPix-NEST-aware multiscale Haar transform (bijective, exact inverse, logdet=0)
# - Starts at NSIDE=64 (Npix=49152) and repeatedly applies 4-ary Haar on NEST quads
# - Returns z (concatenated multiscale coefficients), per-level pieces for visualization,
#   and supports exact inverse back to D with zero reconstruction error (within fp precision).
#
# NOTE: This example uses a fixed orthonormal 4x4 Haar, so the total log-det is exactly 0.
#       It's easy to extend by inserting learned invertible 1x1s or coupling layers per level
#       to get non-zero (tractable) log-dets. Here we keep it minimal & verifiable.

import torch
import math
import matplotlib.pyplot as plt

# ------------------------------
# Core transform
# ------------------------------
class HealpixHaarPyramid(torch.nn.Module):
    def __init__(self, nside_fine: int = 64, device=None, dtype=torch.float32):
        super().__init__()
        assert (nside_fine & (nside_fine - 1)) == 0 and nside_fine >= 1, "NSIDE must be power of 2"
        self.nside_fine = nside_fine
        self.Npix_fine = 12 * nside_fine * nside_fine
        self.level_nsides = []
        n = nside_fine
        while n >= 1:
            self.level_nsides.append(n)
            n //= 2
        # Fixed orthonormal 4x4 Haar-like matrix on quads (NEST children are contiguous)
        W = 0.5 * torch.tensor([
            [ 1.0,  1.0,  1.0,  1.0],  # low-pass (avg * 2)
            [ 1.0,  1.0, -1.0, -1.0],  # detail 1
            [ 1.0, -1.0,  1.0, -1.0],  # detail 2
            [ 1.0, -1.0, -1.0,  1.0],  # detail 3
        ], dtype=dtype)
        # Register as buffers for easy device movement
        self.register_buffer("W", W)          # forward uses @ W.T
        self.register_buffer("W_inv", W.T)    # inverse uses @ W
    
    @property
    def levels(self):
        # number of downsample steps from fine to NSIDE=1
        return len(self.level_nsides) - 1
    
    def forward(self, x):
        """
        x: (B, Npix_fine), NEST ordering assumed.
        Returns:
          z: (B, Npix_fine) concatenation of [coarsest 12 a | details at each level from coarse->fine]
          pieces: dict with per-level tensors for visualization
          logdet: scalar tensor (here zero)
        """
        B, N = x.shape
        assert N == self.Npix_fine, f"Expected N={self.Npix_fine} for NSIDE={self.nside_fine}, got {N}"
        
        v = x
        detail_list = []   # store from fine->coarse as (B, num_parents, 3)
        a_list = []        # store the a (low-pass) at each level (including coarsest 12 at end)
        
        # Iterate fine -> coarse; each step reduces length by 4x
        cur_npix = self.Npix_fine
        for lvl in range(self.levels):
            num_parents = cur_npix // 4
            quads = v.view(B, num_parents, 4)             # (B, P, 4) contiguous NEST children
            y = torch.matmul(quads, self.W.t())           # (B, P, 4); y[...,0]=a, y[...,1:]=details
            a = y[..., 0]                                 # (B, P)
            d = y[..., 1:]                                # (B, P, 3)
            detail_list.append(d)
            a_list.append(a)
            v = a
            cur_npix = num_parents
        
        # v now has shape (B, 12) at NSIDE=1
        a_coarse = v                                       # (B, 12)
        
        # Build z as: [a_coarse | details at levels from coarse->fine]
        # For clarity, also keep per-level pieces to visualize
        z_parts = [a_coarse.reshape(B, -1)]
        # reverse details to start from coarse (closest to a_coarse) to fine
        for d in reversed(detail_list):
            z_parts.append(d.reshape(B, -1))
        z = torch.cat(z_parts, dim=1)                      # (B, Npix_fine)
        
        pieces = {
            "a_coarse": a_coarse,                 # (B, 12)
            "details_coarse_to_fine": list(reversed(detail_list)),  # each (B, P_level, 3)
            "a_fine_to_coarse": a_list,           # each (B, P_level)
            "level_nsides": self.level_nsides
        }
        logdet = torch.zeros(B, dtype=x.dtype, device=x.device)  # orthonormal => logdet 0
        return z, pieces, logdet
    
    def inverse(self, z):
        """
        z: (B, Npix_fine) as produced by forward()
        Returns:
          x_recon: (B, Npix_fine)
          logdet: scalar tensor (zero)
        """
        B, N = z.shape
        assert N == self.Npix_fine, "z must be same length as input"
        
        # Parse z according to the same layout used in forward()
        # We need to know the P (num parents) at each level to slice details
        P_levels = []
        cur = self.Npix_fine
        for _ in range(self.levels):
            P_levels.append(cur // 4)  # parents count at that step
            cur //= 4
        
        # Start by taking a_coarse (12)
        offset = 12
        a = z[:, :offset]  # (B, 12)
        
        # Prepare the detail tensors list (coarse->fine) from z
        details_coarse_to_fine = []
        for P in P_levels[::-1]:  # iterate from coarse to fine when slicing
            # but recall at each coarser level, number of parents is P
            d_size = P * 3
            d_flat = z[:, offset:offset + d_size]
            offset += d_size
            details_coarse_to_fine.append(d_flat.view(B, P, 3))
        
        # Now reconstruct fine map by going coarse->fine
        v = a  # starts at (B, 12)
        for d in details_coarse_to_fine:
            # v is (B, P), d is (B, P, 3)
            y = torch.cat([v.unsqueeze(-1), d], dim=-1)   # (B, P, 4) with [a | details]
            quads = torch.matmul(y, self.W_inv.t())       # inverse: multiply by W (stored as W_inv = W^T)
            v = quads.reshape(B, -1)                      # upsampled to 4*P
        x_recon = v
        logdet = torch.zeros(B, dtype=z.dtype, device=z.device)
        return x_recon, logdet

# ------------------------------
# Demo / verification
# ------------------------------
if __name__ == '__main__':
    device = torch.device("cpu")
    torch.manual_seed(0)

    nside = 64
    Npix = 12 * nside * nside
    B = 3  # small batch for demo

# Create synthetic maps (nontrivial structure): smooth gradient + localized bump + noise
    idx = torch.arange(Npix, dtype=torch.float32).unsqueeze(0).repeat(B,1)
    base = 0.001 * idx  # slow gradient
    noise = torch.randn(B, Npix) * 0.1
    maps = base + noise
# Add a localized feature in one map
    maps[0, 20000:20050] += 3.0

    transform = HealpixHaarPyramid(nside_fine=nside).to(device)

# Forward transform
    z, pieces, logdet = transform(maps)

# Inverse transform
    x_recon, logdet_inv = transform.inverse(z)

# Reconstruction error
    max_abs_err = (x_recon - maps).abs().max().item()

# Energy per scale (sum of squares) to visualize multiscale distribution
    energies = []
    labels = []
# Coarsest a (12)
    energies.append(pieces["a_coarse"].pow(2).sum(dim=1).mean().item())
    labels.append("a(NSIDE=1)")
# Details from coarse->fine
    for lvl, d in enumerate(pieces["details_coarse_to_fine"]):
        e = d.pow(2).sum(dim=(1,2)).mean().item()
        nside_lvl = 2**(lvl)  # since we started at NSIDE=1 and go finer each step
        labels.append(f"d(NSIDE={nside_lvl})")
        energies.append(e)

    print(f"Input shape: {maps.shape}")
    print(f"z shape: {z.shape}")
    print(f"Max |recon-error|: {max_abs_err:.3e}")
    print(f"logdet forward (should be 0): {logdet}")
    print(f"logdet inverse (should be 0): {logdet_inv}")

# ------------------------------
# Visualization
# ------------------------------

# 1) Histogram of coefficients at a mid-level details (pick the 3rd level from coarse side)
    mid_idx = min(2, len(pieces["details_coarse_to_fine"])-1)
    mid_details = pieces["details_coarse_to_fine"][mid_idx][0].reshape(-1).detach().numpy()

    plt.figure()
    plt.hist(mid_details, bins=80)
    plt.title(f"Histogram of detail coeffs (level {mid_idx} from coarse)")
    plt.xlabel("Coefficient value")
    plt.ylabel("Count")
    plt.show()

# 2) Plot energy per scale (coarse a then details from coarse->fine)
    plt.figure()
    plt.bar(range(len(energies)), energies)
    plt.xticks(range(len(energies)), labels, rotation=45, ha="right")
    plt.title("Average energy per scale (batch-mean)")
    plt.ylabel("Sum of squares")
    plt.tight_layout()
    plt.show()

# 3) Visualize first 12 coarsest coefficients for sample 0
# plt.figure()
# plt.stem(pieces["a_coarse"][0].detach().numpy(), use_line_collection=True)
# plt.title("Coarsest 12 coefficients (sample 0)")
# plt.xlabel("Index (base pixels)")
# plt.ylabel("Value")
# plt.show()


# Let's visualize the z coefficients in a more intuitive way:
# For each level's a (low-pass) and each detail channel, reshape to its NSIDE grid
# using the HEALPix nested indexing order.

    import healpy as hp

    def visualize_multiscale_coeffs(pieces, sample_idx=0):
        a_coarse = pieces["a_coarse"][sample_idx]  # (12,)
        details_levels = pieces["details_coarse_to_fine"]  # list of (B, P, 3) from coarse->fine
        level_nsides = pieces["level_nsides"]  # e.g., [64, 32, 16, ..., 1]

        plots = []
        titles = []

        # Coarsest a (NSIDE=1)
        nside_a = 1
        map_a = torch.zeros(12 * nside_a * nside_a)
        map_a[:] = a_coarse
        plots.append(map_a.numpy())
        titles.append(f"a @ NSIDE={nside_a}")

        # Details from coarse->fine
        cur_nside = 1
        for lvl, d in enumerate(details_levels):
            P = d.shape[1]
            nside_parent = cur_nside
            nside_child = nside_parent * 2
            cur_nside = nside_child
            # We have 3 detail channels
            for ch in range(3):
                map_d = torch.zeros(12 * nside_parent * nside_parent)
                map_d[:] = d[sample_idx, :, ch]
                plots.append(map_d.numpy())
                titles.append(f"d{ch+1} @ NSIDE={nside_parent} (parent level)")

        # Plot using mollview
        import matplotlib.pyplot as plt
        n_plots = len(plots)
        plt.figure(figsize=(12, 2*n_plots))
        for i, (m, t) in enumerate(zip(plots, titles), 1):
            hp.mollview(
                m, 
                nest=True, 
                title=t, 
                sub=(n_plots, 1, i), 
                cmap='coolwarm', 
                min=-abs(m).max(), 
                max=abs(m).max(),
                cbar=False
            )
        plt.tight_layout()
        plt.show()

# Visualize for sample 0
    visualize_multiscale_coeffs(pieces, sample_idx=0)
