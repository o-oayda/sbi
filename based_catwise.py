from catwise.maps import CatwiseSim
from tools.priors import DipolePrior
from tools.inference import Inference
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'n_workers',
    type=int,
    help='Number of workers to distribute simuation over.'
)
args = parser.parse_args()

N_SIM = 20
N_WORKERS = args.n_workers

sim = CatwiseSim(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()
prior = DipolePrior(
    mean_count_range=[25_000_000, 35_000_000],
    amplitude_range=[0, 0.01]
)

inferer = Inference(prior, sim)
inferer.make_batch_simulations(
    n_simulations=N_SIM,
    n_workers=N_WORKERS,
    save=True,
    custom_save_dir='catwise_0p5_17p0'
)