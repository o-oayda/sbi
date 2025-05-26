import torch

SIM_DIR = 'catwise_0p5_17p0'

theta, x = torch.load(f'simulations/{SIM_DIR}/theta_and_x_nan.pt')
mask = torch.isnan(x)
x[mask] = 0
torch.save((theta, x), f'simulations/{SIM_DIR}/theta_and_x.pt')