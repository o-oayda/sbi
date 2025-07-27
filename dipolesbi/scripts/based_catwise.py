from dipolesbi.catwise.maps import CatwiseSim
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.inference import Inference
from sbi.utils import BoxUniform
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'n_simulations',
    type=int,
    help='Number of simulations to run.'
)
parser.add_argument(
    'n_workers',
    type=int,
    help='Number of workers to distribute simuation over.'
)
parser.add_argument(
    'save_dir',
    type=str,
    help='Name of directory in simulations to save into (automatically created).'
)
args = parser.parse_args()

N_SIM = args.n_simulations
N_WORKERS = args.n_workers
SAVE_DIR = args.save_dir

sim = CatwiseSim(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()
prior = DipolePrior(
    mean_count_range=[30_000_000, 40_000_000],
    amplitude_range=[0, 0.01]
)
prior.add_prior( # eta_w1
    prior=BoxUniform(
        low=torch.ones(1),
        high=3 * torch.ones(1)
    ),
    index=1
)
prior.add_prior( # eta_w2
    prior=BoxUniform(
        low=torch.ones(1),
        high=3 * torch.ones(1)
    ),
    index=2
)

inferer = Inference(prior, sim)
inferer.make_batch_simulations(
    n_simulations=N_SIM,
    n_workers=N_WORKERS,
    save=True,
    custom_save_dir=SAVE_DIR
)