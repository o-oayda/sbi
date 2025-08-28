from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from jax import numpy as jnp


@dataclass
class TrainingConfig:
    '''
    Configuration for multi-round NLE training.
    '''
    learning_rate: float = 5e-4
    max_n_iter: int = 1000
    batch_size: int = 100
    patience: int = 20

@dataclass
class NLEConfig:
    '''
    Configuration for the NLE.
    '''
    decoder_distribution: Literal['gaussian', 'poisson', 'nb', 'students_t'] = 'gaussian'
    decoder_n_neurons: int = 64
    decoder_n_layers: int = 3
    conditioner: Literal['made', 'mlp', 'transformer'] = 'made'
    conditioner_n_neurons: int = 128
    conditioner_n_layers: int = 3

    n_layers: Optional[int] = 3 # won't apply to heirarchical
    permute_data: bool = False # only for coarse
    maf_stack_size: Optional[int] = None # only for heirarchical
    n_coarse: int = 0 # only for coarse
    blocks: list[tuple[jnp.ndarray, int, int]] = [] # only for heirarchical
    data_reduction_factor: Optional[float] = None # only for standard/coarse

    blocks: list[tuple[jnp.ndarray, int, int]] = field(default_factory=list)
    conditioner_kwargs: dict = field(default_factory=dict)
    flow_type: str = field(default='custom', init=False)

    @classmethod
    def heirarchical(
        cls,
        maf_stack_size: int = 8,
    ) -> 'NLEConfig':
        """Coarse-grained flow configuration for faster training."""
        config = cls(
            maf_stack_size=maf_stack_size,
        )
        config.flow_type = 'heirarchical'
        return config

    @classmethod
    def standard(
        cls,
        n_layers: int = 3,
        data_reduction_factor = 0.5
    ) -> 'NLEConfig':
        """Coarse-grained flow configuration for faster training."""
        config = cls(
            n_layers=n_layers,
            data_reduction_factor=data_reduction_factor
        )
        config.flow_type = 'standard'
        return config

    @classmethod
    def coarse(
        cls
    ) -> 'NLEConfig':
        """Coarse-grained flow configuration for faster training."""
        raise NotImplementedError

