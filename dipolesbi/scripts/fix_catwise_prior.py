from dipolesbi.tools.priors import DipolePrior
from sbi.utils import BoxUniform
import torch
from sbi.utils.user_input_checks import process_prior
import pickle

prior = DipolePrior(
    mean_count_range=[30_000_000, 40_000_000],
    speed_range=[0, 0.01]
)
prior.add_prior(
    prior=BoxUniform(
        low=torch.ones(1),
        high=3 * torch.ones(1)
    ),
    index=1
)
prior.to('cpu')

prior, num_parameters, prior_returns_numpy = process_prior(
    prior, # type: ignore
    custom_prior_wrapper_kwargs={
        'lower_bound': torch.as_tensor(
            prior.low_ranges, device=prior.device
        ),
        'upper_bound': torch.as_tensor(
            prior.high_ranges, device=prior.device
        )
    }
)

custom_save_dir = 'catwise_0p5_17p0_error_scale'
base_path = f'simulations/{custom_save_dir}'
prior_path = f'{base_path}/prior.pkl'

print(f'Saving prior to {prior_path}...')
with open(prior_path, "wb") as handle:
    pickle.dump(prior, handle)
