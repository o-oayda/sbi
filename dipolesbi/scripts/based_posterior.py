from dipolesbi.tools.inference import Inference
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

inference = Inference()
inference.run_sbi(
    sim_dir=SAVE_DIR,
    device=DEVICE,
    load_simulations_in_vram=LOAD_IN_VRAM,
    simulation_fraction=SIM_FRACTION
)
inference.save_posterior(f'based_posterior_{SAVE_DIR}.pkl')
