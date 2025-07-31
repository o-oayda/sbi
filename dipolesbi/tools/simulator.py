from typing import Callable
from numpy.typing import NDArray
from sbi.inference import simulate_for_sbi
from sbi.inference.trainers.base import check_sbi_inputs
from sbi.utils import process_simulator
from sbi.utils.user_input_checks import CustomPriorWrapper
from torch import Tensor
from torch.distributions.distribution import Distribution
import os
import pickle
import torch
import numpy as np


class Simulator:
    def __init__(
            self,
            sbi_processed_prior: CustomPriorWrapper,
            simulation_model: Callable[..., NDArray | Tensor]
    ) -> None:
        self.theta: Tensor | None = None
        self.x: Tensor | None = None
        self.simulation_model = simulation_model
        self.sbi_proposal_distribution = sbi_processed_prior

    def _interface_with_simulator(self, Theta: Tensor) -> NDArray | Tensor:
        mapping = {
            kwarg: np.float64(Theta[i]) for i, kwarg in enumerate(
                self.sbi_proposal_distribution.custom_prior.simulator_kwargs
            )
        } 
        return self.simulation_model(**mapping)

    def _batch_simulator(self,
            prior_returns_numpy: bool,
            n_samples: int,
            n_workers: int
    ) -> None:
        # self.make_simulation can return NDArrays as process_simulator will
        # wrap function with conversion torch tensors (probable slowdown though)
        sbi_processed_simulator = process_simulator(
            self._interface_with_simulator,
            self.sbi_proposal_distribution, 
            prior_returns_numpy
        )
        check_sbi_inputs(
            simulator=sbi_processed_simulator,
            prior=self.sbi_proposal_distribution
        )
        self.theta, self.x = simulate_for_sbi(
            simulator=sbi_processed_simulator,
            proposal=self.sbi_proposal_distribution,
            num_workers=n_workers,
            num_simulations=n_samples
        )

    def _simulation_is_loaded(self):
        if (
               (self.x is None)
            or (self.theta is None)
            or (self.sbi_proposal_distribution is None)
        ):
            return False
        else:
            return True

    def save_simulation(self,
            proposal_distribution: Distribution,
            custom_save_dir: str | None = None
    ) -> None:
        assert self.theta is not None and self.x is not None, (
            'Theta and x have not been created. Run or load a simulation.'
        )
        
        if custom_save_dir is None:
            if not os.path.exists('simulations/'):
                os.makedirs('simulations/')
            i = 1
            while os.path.exists(f'simulations/sim{i}/'):
                i += 1
            base_path = f'simulations/sim{i}'
        else:
            base_path = f'simulations/{custom_save_dir}'
       
        os.makedirs(f'{base_path}/')
        simulation_path = f'{base_path}/theta_and_x.pt'
        prior_path = f'{base_path}/prior.pkl'

        print(f'Saving theta and x to {simulation_path}...')
        torch.save([self.theta, self.x], simulation_path)

        print(f'Saving prior to {prior_path}...')
        with open(prior_path, "wb") as handle:
            pickle.dump(proposal_distribution, handle)

    def load_simulation(self, sim_dir: str) -> None:
        if not os.path.exists(f'simulations/{sim_dir}/'):
            raise FileNotFoundError(f'Cannot find {sim_dir}.')
        
        print(f'Opening {sim_dir}...')
        sim_path = f'simulations/{sim_dir}/theta_and_x.pt'
        self.theta, self.x = torch.load(sim_path)
        
        prior_path = f'simulations/{sim_dir}/prior.pkl'
        print(f'Opening {prior_path}...')
        with open(prior_path, "rb") as handle:
            self.sbi_proposal_distribution = pickle.load(handle)
