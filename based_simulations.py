from tools.maps import SkyMap
from tools.priors import DipolePrior
import argparse
from tools.inference import Inference

parser = argparse.ArgumentParser()
parser.add_argument(
    'n_workers',
    type=int,
    help='Number of workers to distribute simuation over.'
)
args = parser.parse_args()

N_SIM = 20
N_WORKERS = args.n_workers

sim = SkyMap()
sim.configure(
    dipole_method='base',
    dipole_hyperparameters={
        'flux_percentage_noise': 'ecliptic',
        'minimum_flux_cut': 3
    }
)
prior = DipolePrior(
    mean_count_range=[8_000_000, 12_000_000],
    amplitude_range=[0, 0.01]
)

inferer = Inference(prior, sim)
inferer.make_batch_simulations(
    n_simulations=N_SIM,
    n_workers=N_WORKERS,
    save=True
)