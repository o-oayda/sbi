from dipolesbi.tools.inference import LikelihoodFreeInferer
from dipolesbi.tools.simulator import Simulator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	'save_dir',
	type=str,
	help='Name of directory to load simulations from.'
)
parser.add_argument(
	'--device',
	type=str,
	choices=['cpu', 'cuda'],
	default='cuda',
	help="Device to run inference on ('cpu' or 'cuda')."
)
parser.add_argument(
	'--load_in_vram',
	action='store_true',
	help='If set, load simulations into VRAM (default: False).'
)
parser.add_argument(
	'--simulation_fraction',
	type=float,
	default=1.0,
	help='Fraction of data to use for simulating (default: 1.0).'
)
args = parser.parse_args()

SAVE_DIR = args.save_dir
DEVICE = args.device
LOAD_IN_VRAM = args.load_in_vram
SIM_FRACTION = args.simulation_fraction

sim = Simulator(path_to_saved_simulations=SAVE_DIR)
inference = LikelihoodFreeInferer(sim)
inference.run_healpix_sbi(
    training_device=DEVICE,
    load_simulations_in_vram=LOAD_IN_VRAM,
    simulation_fraction=SIM_FRACTION
)
inference.save_posterior(f'based_posterior_{SAVE_DIR}.pkl')
