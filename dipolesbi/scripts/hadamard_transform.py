from dipolesbi.tools.transforms import HaarWaveletTransform
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from dipolesbi.tools.np_rngkey import prng_key


NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
NBATCH = 1000
key = prng_key(42)

dmap1 = key.poisson(50, shape=(NPIX,))[None, :]
transform1 = HaarWaveletTransform(first_nside=NSIDE, last_nside=1, post_normalise=False)
zmap1, _ = transform1(dmap1)
zmap1 = zmap1.squeeze()

lam = 50 * np.ones((NBATCH, NPIX))
dmap2 = key.poisson(lam=lam, shape=lam.shape)
transform2 = HaarWaveletTransform(first_nside=NSIDE, last_nside=1, post_normalise=True)
zmap2, _ = transform2(dmap2)
zmap2 = zmap2.squeeze()
