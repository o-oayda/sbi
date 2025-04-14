# %%
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from funcs import simulation, PolarPrior, DipolePoisson
from sbi.inference import NPE, NLE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.utils import BoxUniform
import torch
from sbi.analysis import pairplot
from corner import corner
# %%
D = 0.05
PHI =  5
THETA = 1
NBAR = 50

dmap = simulation([NBAR, D, PHI, THETA])
hp.projview(dmap)
plt.show()
# %%
model = DipolePoisson(dmap)
model.run_dynesty()
corner(model.dresults.samples_equal())
plt.show()

model.run_sbi()
corner(model.samples)
plt.show()