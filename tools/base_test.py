# %%
from maps import SkyMap
import torch
import healpy as hp
# %%

model = SkyMap()
model.generate_dipole_from_base(
    observer_direction=(5, 1),
    n_initial_points=1_000_000
)
model.mask_pixels(fill_value=0)

hp.projview(model.density_map.numpy())
# %%