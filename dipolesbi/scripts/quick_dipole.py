import argparse
from dipolesbi.catwise.maps import Catwise
import time
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.plotting import smooth_map
import matplotlib.pyplot as plt
from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.utils import batch_simulate


parser = argparse.ArgumentParser()
parser.add_argument(
    '--workers',
    type=int,
    help='Number of workers to distribute sim jobs over.'
)
parser.add_argument(
    '--sims',
    type=int,
    help='Number of simulations to generate.'
)
args = parser.parse_args()

sim = Catwise(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()

key = prng_key(42)
prior = DipolePriorNP(mean_count_range=[30e6, 40e6])
theta = prior.sample(key, n_samples=args.sims)

t0 = time.time()
sims, masks = batch_simulate(theta, sim.generate_dipole, n_workers=args.workers)
t1 = time.time()
print(sims.dtype)

dt = t1 - t0
dt_per_sim = dt / args.sims
print(f'Time (total): {dt:.3g} s')
print(f'Time (per sim): {dt_per_sim:.3g} s')

smooth_map(sims[0, :] * masks[0, :])
plt.savefig('memtest_out.png')
