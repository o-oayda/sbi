from dipolesbi.tools.priors import DipolePrior
from sbi.utils import BoxUniform
import torch
from sbi.utils.user_input_checks import process_prior
import pickle

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

prior, num_parameters, prior_returns_numpy = process_prior(
    prior, # type: ignore
    custom_prior_wrapper_kwargs={
        'lower_bound': prior.low_ranges,
        'upper_bound': prior.high_ranges 
    }
)

custom_save_dir = 'catwise_0p5_17p0_error_scale'
base_path = f'simulations/{custom_save_dir}'
prior_path = f'{base_path}/prior.pkl'

print(f'Saving prior to {prior_path}...')
with open(prior_path, "wb") as handle:
    pickle.dump(prior, handle)
