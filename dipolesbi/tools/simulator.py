from typing import Callable, Optional, cast
from numpy.typing import NDArray
from sbi.inference import simulate_for_sbi
from sbi.inference.trainers.base import check_sbi_inputs
from sbi.utils import process_simulator
from sbi.utils.user_input_checks import CustomPriorWrapper, process_prior
from torch import Tensor
import os
import pickle
import torch
from dipolesbi.tools import Prior


class Simulator:
    def __init__(
            self,
            prior: Optional[Prior] = None,
            simulation_model: Optional[Callable[..., NDArray | Tensor]] = None,
            path_to_saved_simulations: Optional[str] = None
    ) -> None:
        self.theta: Tensor | None = None
        self.x: Tensor | None = None
        
        if path_to_saved_simulations is not None:
            assert (prior is None) and (simulation_model is None), (
                'Either specify a path to simulations or a prior and model.'
            )
            self.load_simulation(path_to_saved_simulations)
        else:
            assert path_to_saved_simulations is None, (
                'Either specify a path to simulations or a prior and model.'
            )
            assert prior is not None and simulation_model is not None
            self.simulation_model = simulation_model
            self.prior = prior
            self.sbi_processed_prior = self._process_prior()

    def _interface_with_simulator(self, Theta: Tensor) -> NDArray | Tensor:
        mapping = {
            kwarg: Theta[..., i].numpy() for i, kwarg in enumerate(
                self.sbi_processed_prior.custom_prior.simulator_kwargs
            ) # ... ensures we slice down the parameter axis; ok for 1 dim arrays
        } 
        return self.simulation_model(**mapping)

    def _process_prior(self) -> CustomPriorWrapper:
        sbi_processed_prior, *_ = process_prior(self.prior, #type: ignore
            custom_prior_wrapper_kwargs={
                'lower_bound': self.prior.low_ranges,    
                'upper_bound': self.prior.high_ranges
            }
        )
        # since we pass a custom Prior from prior.py,
        # this should be a CustomPriorWrapper
        sbi_processed_prior = cast(CustomPriorWrapper, sbi_processed_prior)
        return sbi_processed_prior

    def make_batch_simulations(self,
            n_simulations: int,
            n_workers: int,
            **kwargs
    ) -> tuple[Tensor, Tensor]:
        # self.make_simulation can return NDArrays as process_simulator will
        # wrap function with conversion torch tensors (probable slowdown though)
        sbi_processed_simulator = process_simulator(
            self._interface_with_simulator,
            self.sbi_processed_prior, 
            is_numpy_simulator=False
        )
        check_sbi_inputs(
            simulator=sbi_processed_simulator,
            prior=self.sbi_processed_prior
        )
        self.theta, self.x = simulate_for_sbi(
            simulator=sbi_processed_simulator,
            num_workers=n_workers,
            num_simulations=n_simulations,
            **{
                'proposal': self.prior,
                **kwargs # possibility of overwriting prior proposal in kwargs
            }
        )
        print(f'Made {n_simulations} simulations.')
        return self.theta, self.x

    def _simulation_is_loaded(self):
        if (
               (self.x is None)
            or (self.theta is None)
            or (self.prior is None)
        ):
            return False
        else:
            return True

    def save_simulation(self,
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
            if os.path.exists(base_path):
                suffix = "_copy"
                count = 1
                new_base_path = f"{base_path}{suffix}"
                while os.path.exists(new_base_path):
                    count += 1
                    new_base_path = f"{base_path}{suffix}{count}"
                base_path = new_base_path
       
        os.makedirs(f'{base_path}/')
        simulation_path = f'{base_path}/theta_and_x.pt'
        prior_path = f'{base_path}/prior.pkl'
        prior_info_path = f'{base_path}/prior_info.txt'

        print(f'Saving theta and x to {simulation_path}...')
        torch.save([self.theta, self.x], simulation_path)

        print(f'Saving prior to {prior_path}...')
        self.sbi_processed_prior.to('cpu') # so I don't get fucked later
        with open(prior_path, "wb") as handle:
            pickle.dump(self.sbi_processed_prior, handle)

        print(f'Saving prior info to {prior_info_path}...')
        self.sbi_processed_prior.custom_prior.write_prior_info(prior_info_path)

    def load_simulation(self, sim_dir: str) -> None:
        if not os.path.exists(f'simulations/{sim_dir}/'):
            raise FileNotFoundError(f'Cannot find {sim_dir}.')
        
        print(f'Opening {sim_dir}...')
        sim_path = f'simulations/{sim_dir}/theta_and_x.pt'
        self.theta, self.x = torch.load(sim_path)
        
        prior_path = f'simulations/{sim_dir}/prior.pkl'
        print(f'Opening {prior_path}...')
        with open(prior_path, "rb") as handle:
            self.sbi_processed_prior = pickle.load(handle)
