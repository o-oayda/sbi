from healpy import nside2npix
import torch
import matplotlib.pyplot as plt
from dipolesbi.tools.transforms import HealpixSOPyramid, learn_transformation
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cpu")
dtype = torch.float32

def make_batch(batch_size: int, Npix: int):
    idx = torch.arange(Npix, dtype=dtype).unsqueeze(0).repeat(batch_size,1)
    base = 0.001 * idx
    noise = torch.randn(batch_size, Npix, dtype=dtype) * 0.2
    maps = base + noise
    for b in range(batch_size):
        start = (b+1)*500
        maps[b, start:start+40] += 2.0
    return maps

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    nside = 16
    model = HealpixSOPyramid(nside_fine=nside)
    data = make_batch(50_000, 12 * nside * nside)

    # Pretrain Q_l to increase Gaussianity (diagonal NLL)
    # losses = pretrain(model, data, steps=500, batch_size=64, lr=1e-3)
    model, history, validation = learn_transformation(model, data, epochs=3)

    # Check exact invertibility
    D = make_batch(4, nside2npix(nside))
    z, *_ = model(D)
    D_rec, *_ = model.inverse(z)
    recon_err = (D - D_rec).abs().max().item()
    print(f"Max |reconstruction error|: {recon_err:.3e}")

    # Plot training curve
    plt.figure()
    plt.plot(history['train'])
    plt.xlabel("step")
    plt.ylabel("diag-Gaussian NLL")
    plt.title("SO(4) pretraining loss")
    plt.tight_layout(); plt.show()

    # Quick intuition plot: Haar vs learned for a mid-level
    # Build Haar once for comparison (not used in the model)
    W = 0.5 * torch.tensor([
        [ 1.0,  1.0,  1.0,  1.0],
        [ 1.0,  1.0, -1.0, -1.0],
        [ 1.0, -1.0,  1.0, -1.0],
        [ 1.0, -1.0, -1.0,  1.0],
    ], dtype=dtype)

    def get_level_quads_haar(D, levels, target_level_from_coarse):
        # Build quads fine->coarse by grouping, then apply Haar at the desired level
        v = D
        cur = D.shape[1]
        quads_f2c = []
        for _ in range(levels):
            P = cur // 4
            quads = v.view(D.shape[0], P, 4)
            quads_f2c.append(quads)
            v = quads.mean(dim=-1)  # just mean to propagate; this is only for a baseline view
            cur = P
        idx = levels - 1 - target_level_from_coarse  # map coarse idx -> fine idx
        quads = quads_f2c[idx]
        y = torch.matmul(quads, W.T)
        return y

    # Gather samples
    D = make_batch(64, nside2npix(nside))
    _, per_level_quads_after, *_ = model(D)   # rotated by learned SO(4), coarse->fine order
    mid_level = min(2, model.levels-1)       # pick a mid level (0=coarsest)
    y_after = per_level_quads_after[mid_level].reshape(-1, 4).detach()  # (B*P,4)
    y_haar = get_level_quads_haar(D, model.levels, mid_level).reshape(-1, 4).detach()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(y_haar[:,1].numpy(), bins=200)
    plt.title(f"Haar d1 (level {mid_level})")
    plt.subplot(1,2,2)
    plt.hist(y_after[:,1].numpy(), bins=80)
    plt.title(f"Learned SO(4) d1 (level {mid_level})")
    plt.tight_layout()
    plt.show()
