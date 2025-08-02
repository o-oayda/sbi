from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.inference import LikelihoodFreeInferer
from dipolesbi.tools.simulator import Simulator
from sbi.utils import BoxUniform
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--n_simulations',
    type=int,
    help='Number of simulations to run.'
)
parser.add_argument(
    '--n_workers',
    type=int,
    help='Number of workers to distribute simuation over.'
)
parser.add_argument(
    '--save_dir',
    type=str,
    help='Name of directory in simulations to save into (automatically created).'
)
parser.add_argument(
    '--error_dist',
    type=str,
    default='gaussian',
    help='Error distribution to use for CatWISE errors (gaussian or students-t; default: gaussian).'
)
parser.add_argument(
    '--use_float32',
    action='store_true',
    help='If set, use float64 for internal calculations but convert to float32 after.'
)
args = parser.parse_args()

N_SIM = args.n_simulations
N_WORKERS = args.n_workers
SAVE_DIR = args.save_dir
ERROR_DIST = args.error_dist
USE_FLOAT32 = args.use_float32

model = Catwise(
    cat_w1_max=17.0, 
    cat_w12_min=0.5,
    magnitude_error_dist=ERROR_DIST,
    use_float32=USE_FLOAT32
)
model.initialise_data()
prior = DipolePrior(
    mean_count_range=[30_000_000, 40_000_000],
    speed_range=[0, 8]
)
prior.add_prior(
    prior=BoxUniform(
        low= 0 * torch.ones(1),
        high=8 * torch.ones(1)
    ),
    short_name='etaW1',
    simulator_kwarg='w1_extra_error',
    index=1
)
prior.add_prior(
    prior=BoxUniform(
        low= 0 * torch.ones(1),
        high=8 * torch.ones(1)
    ),
    short_name='etaW2',
    simulator_kwarg='w2_extra_error',
    index=2
)
prior.add_prior(
    prior=BoxUniform(
        low=-1 * torch.ones(1),
        high=3 * torch.ones(1)
    ),
    short_name='nu',
    simulator_kwarg='log10_magnitude_error_shape_param',
    index=3
)

simulator = Simulator(prior, model.generate_dipole)
simulator.make_batch_simulations(
    n_simulations=N_SIM,
    n_workers=N_WORKERS,
)
simulator.save_simulation(SAVE_DIR)
