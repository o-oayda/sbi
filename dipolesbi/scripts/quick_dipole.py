import argparse
from typing import Callable

from joblib import Parallel, delayed
from dipolesbi.catwise.maps import Catwise
import time
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.plotting import smooth_map
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
from tqdm import tqdm
from dipolesbi.tools.priors_np import DipolePriorNP


def batch_simulate(
        theta: dict[str, NDArray],
        model_callable: Callable[..., NDArray],
        n_workers: int
) -> NDArray:
    simulation_batch_size = 1
    n_simulations = list(theta.values())[0].shape[0]
    n_batches = n_simulations // simulation_batch_size # batch size of 1 by default in simulate_for_sbi

    theta_batches = [
        {key: arr_batch for key, arr_batch in zip(theta.keys(), batch)}
        for batch in zip(*[
            np.array_split(arr, n_batches, axis=0)
            for arr in theta.values()
        ])
    ]

    simulation_outputs: list[NDArray] = [ # type: ignore
        xx
        for xx in tqdm(
            Parallel(return_as='generator', n_jobs=n_workers)(
                delayed(model_callable)(**batch)
                for batch in theta_batches
            ),
            total=n_simulations
        )
    ]

    x = np.vstack(simulation_outputs)
    return x


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
sims = batch_simulate(theta, sim.generate_dipole, n_workers=args.workers)
t1 = time.time()

dt = t1 - t0
dt_per_sim = dt / args.sims
print(f'Time (total): {dt:.3g} s')
print(f'Time (per sim): {dt_per_sim:.3g} s')

smooth_map(sims[0, :])
plt.savefig('memtest_out.png')
