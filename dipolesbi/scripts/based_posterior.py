from dipolesbi.tools.inference import Inference
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	'save_dir',
	type=str,
	help='Name of directory to load simulations from.'
)
args = parser.parse_args()

SAVE_DIR = args.save_dir

inference = Inference()
inference.run_sbi(sim_dir=SAVE_DIR)
inference.save_posterior(f'based_posterior_{SAVE_DIR}.pkl')
