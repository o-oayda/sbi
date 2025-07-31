from abc import ABC, abstractmethod
from torch import Tensor
from torch.distributions.distribution import Distribution
import os
import pickle
import torch


class Simulator(ABC):
    def __init__(self) -> None:
        self.theta: Tensor | None = None
        self.x: Tensor | None = None

    @abstractmethod
    def batch_simulator(self,
            proposal_distribution: Distribution,
            n_samples: int,
            n_workers: int
    ) -> None:
        pass

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
