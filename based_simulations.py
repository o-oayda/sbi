from tools.maps import SkyMap
from tools.models import DipolePoisson
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'n_workers',
    type=int,
    help='Number of workers to distribute simuation over.'
)
args = parser.parse_args()

N, V, PHI, THETA = [10_000_000, 0.009, 5, 1]
N_SIM = 50
N_WORKERS = args.n_workers

# generate example sky map
sim = SkyMap()
sim.generate_dipole_from_base(
    observer_direction=(PHI, THETA),
    n_initial_points=N,
    observer_speed=V
)
sim.mask_pixels(fill_value=0)

# instantiate model and run simulation
model = DipolePoisson(
    sim.density_map,
    mean_count_range=[8_000_000, 12_000_000],
    amplitude_range=[0, 0.01]
)
model.make_batch_simulations(
    n_simulations=N_SIM,
    n_workers=N_WORKERS,
    dipole_method='base',
    save=True
)