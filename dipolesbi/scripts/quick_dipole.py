from dipolesbi.catwise.maps import CatwiseSim

sim = CatwiseSim(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()
sim.generate_dipole(n_initial_samples=35_000_000)
