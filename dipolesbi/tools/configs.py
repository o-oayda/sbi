from dataclasses import asdict, dataclass, field
from typing import Optional, Literal
from numpy.typing import NDArray
from dipolesbi.tools.transforms import BlankTransform, HaarWaveletTransform, InvertibleDataTransform, ZScore


@dataclass
class TrainingConfig:
    '''
    Configuration for multi-round NLE training.
    '''
    learning_rate: float = 5e-4
    max_n_iter: int = 1000
    min_n_iter: int = 2
    batch_size: int = 100
    patience: int = 20
    # adam_b2: float = 0.999
    restore_from_previous: bool = False
    warmup_epochs: float = 0.3
    min_lr_ratio: float = 0.1
    clip_norm: float = 1.
    weight_decay: float = 1e-5
    shuffle_train: bool = True
    shuffle_val: bool = False
    validation_fraction: float = 0.1
    weight_by_round: bool = False
    alpha_weight: float = 1.

@dataclass
class SurjectiveNLEConfig:
    '''
    Configuration for the NLE.
    '''
    surjective_layer_type: Literal[
        'affine_MAF', 'rational_quadratic_MAF'
    ] = 'affine_MAF'
    decoder_distribution: Literal['gaussian', 'poisson', 'nb', 'students_t'] = 'gaussian'
    decoder_n_neurons: int = 64
    decoder_n_layers: int = 3
    conditioner: Literal['made', 'mlp', 'transformer'] = 'made'
    conditioner_n_neurons: int = 128
    conditioner_n_layers: int = 3

    n_layers: Optional[int] = None # won't apply to heirarchical
    permute_data: bool = False # only for coarse
    maf_stack_size: Optional[int] = None # only for heirarchical
    n_coarse: Optional[int] = 0 # only for coarse
    data_reduction_factor: Optional[float] = None # only for standard/coarse

    conditioner_kwargs: dict = field(default_factory=dict)
    flow_type: str = field(default='custom', init=False)

    def __post_init__(self):
        if self.surjective_layer_type in ['affine_MAF', 'rational_quadratic_MAF']:
            assert self.conditioner == 'made', (
                f'Conditioner must be MADE when using {self.surjective_layer_type}.'
            )

    @classmethod
    def heirarchical(
        cls,
        maf_stack_size: int = 8,
        **overrides
    ) -> 'SurjectiveNLEConfig':
        """Coarse-grained flow configuration for faster training."""
        config = cls(
            maf_stack_size=maf_stack_size,
            **overrides
        )
        config.flow_type = 'heirarchical'
        return config

    @classmethod
    def one_and_done(
        cls,
        maf_stack_size: int = 8,
        **overrides
    ) -> 'SurjectiveNLEConfig':
        """Coarse-grained flow configuration for faster training."""
        config = cls(
            maf_stack_size=maf_stack_size,
            **overrides
        )
        config.flow_type = 'one_and_done'
        return config

    @classmethod
    def standard(
        cls,
        n_layers: int = 3,
        data_reduction_factor = 0.5,
        **overrides
    ) -> 'SurjectiveNLEConfig':
        """Coarse-grained flow configuration for faster training."""
        config = cls(
            n_layers=n_layers,
            data_reduction_factor=data_reduction_factor,
            **overrides
        )
        config.flow_type = 'standard'
        return config

    @classmethod
    def coarse(
        cls
    ) -> 'SurjectiveNLEConfig':
        """Coarse-grained flow configuration for faster training."""
        raise NotImplementedError

@dataclass
class MultiRoundInfererConfig:
    '''
    Configuration for the multi-round inferer.
    '''
    simulation_budget: int
    n_rounds: int
    prng_integer_seed: int = 42
    load_simulations: Optional[str] = None
    reference_theta: Optional[dict[str, NDArray]] = None
    plot_save_dir: str = 'nle_out'
    simulations_per_round: int = field(init=False)
    simulation_path: Optional[str] = field(init=False)
    dequantise_data: bool = False
    n_requantisations: Optional[int] = None
    initial_fraction: float = 0.3
    n_likelihood_samples: int = 50_000

    def __post_init__(self) -> None:
        self.simulations_per_round = self.simulation_budget // self.n_rounds
        
        if self.load_simulations is not None:
            self.simulation_path = f'{self.plot_save_dir}/{self.load_simulations}/data'
        else:
            self.simulation_path = None

        if self.dequantise_data:
            assert self.n_requantisations is not None, (
                'Supply number of requantisations when setting dequantise_data=True.'
            )

@dataclass
class TransformConfig:
    transform: InvertibleDataTransform = field(default=BlankTransform(), init=False)
    first_nside: Optional[int] = None
    last_nside: Optional[int] = None
    post_normalise: Optional[bool] = None
    matrix_type: Optional[str] = None
    normalise_details: Optional[bool] = None
    n_chunks: Optional[int] = None

    @classmethod
    def haar_wavelet(
            cls,
            first_nside: int = 16,
            last_nside: int = 1,
            post_normalise: bool = False,
            matrix_type: str = 'hadamard',
            normalise_details: bool = True,
            n_chunks: int = 1
    ) -> 'TransformConfig':
        config = cls(
            first_nside=first_nside,
            last_nside=last_nside,
            post_normalise=post_normalise,
            matrix_type=matrix_type,
            normalise_details=normalise_details,
            n_chunks=n_chunks
        )
        config_dict = asdict(config)
        del config_dict['transform']
        data_transform_instance = HaarWaveletTransform(**config_dict)
        config.transform = data_transform_instance
        return config

    @classmethod
    def blank_transform(cls) -> 'TransformConfig':
        config = cls()
        config.transform = BlankTransform()
        return config
        
@dataclass
class ConfigOfConfigs:
    '''
    Meta configuration class for configs which, from experimenting, work well in
    particular settings. Access each setting by calling the relevant class method.
    '''
    training_config: TrainingConfig
    multiround_config: MultiRoundInfererConfig
    ssnle_config: SurjectiveNLEConfig
    transform_config: TransformConfig

    @classmethod
    def blank(
            cls,
            reference_theta: dict[str, NDArray],
            training_overrides: dict = {},
            multiround_overrides: dict = {},
            ssnle_overrides: dict = {}
    ) -> 'ConfigOfConfigs':
        transform_config = TransformConfig.blank_transform()
        train_config = TrainingConfig(**training_overrides)
        mr_config_dict = {
            'simulation_budget': 10_000,
            'n_rounds': 15,
            **multiround_overrides
        }
        mr_config = MultiRoundInfererConfig(
            reference_theta=reference_theta, 
            **mr_config_dict
        )
        nle_config = SurjectiveNLEConfig.standard(**ssnle_overrides)
        return cls(
            training_config=train_config,
            multiround_config=mr_config,
            ssnle_config=nle_config,
            transform_config=transform_config
        )

    # low learning rate high nside?
    # ok not weighting by round helps in keeping posterior narrow
    @classmethod
    def nside16(
            cls,
            reference_theta: dict[str, NDArray],
            training_overrides: dict = {},
            multiround_overrides: dict = {},
            ssnle_overrides: dict = {},
            transform_overrides: dict = {},
            flow_type_override: Optional[Literal['one_and_done']] = None
    ) -> 'ConfigOfConfigs':
        transform_dict = {
            'first_nside': 16,
            'last_nside': 1,
            'post_normalise': False,
            'matrix_type': 'hadamard',
            'normalise_details': True,
            'n_chunks': 1,
            **transform_overrides
        }
        train_dict = {
            'patience': 20, 
            'learning_rate': 5e-5, # 5e-5 for optimal nside=16
            'restore_from_previous': True,
            'weight_by_round': False,
            **training_overrides
        }
        mr_dict = {
            'simulation_budget': 50_000,
            'n_rounds': 15,
            'reference_theta': reference_theta,
            'dequantise_data': False,
            'initial_fraction': 0.5,
            'n_likelihood_samples': 25_000,
            **multiround_overrides
        }
        nle_dict = {
            'maf_stack_size': 15,
            'conditioner_n_layers': 4,
            'conditioner_n_neurons': 256,
            'decoder_n_layers': 3,
            'decoder_n_neurons': 64, # keep decoder_n_neurons ~ 64 for nside=16?
            'decoder_distribution': 'gaussian',
            **ssnle_overrides
        }
        train_config = TrainingConfig(**train_dict)
        mr_config = MultiRoundInfererConfig(**mr_dict)

        if flow_type_override:
            type_to_method = {'one_and_done': SurjectiveNLEConfig.one_and_done}
            nle_config = type_to_method[flow_type_override](**nle_dict)
        else:
            nle_config = SurjectiveNLEConfig.heirarchical(**nle_dict)

        transform_config = TransformConfig.haar_wavelet(**transform_dict)
        return cls(
            training_config=train_config,
            multiround_config=mr_config,
            ssnle_config=nle_config,
            transform_config=transform_config
        )

    @classmethod
    def nside32(
            cls,
            reference_theta: dict[str, NDArray],
            training_overrides: dict = {},
            multiround_overrides: dict = {},
            ssnle_overrides: dict = {},
            transform_overrides: dict = {}
    ) -> 'ConfigOfConfigs':
        transform_dict = {
            'first_nside': 32,
            'last_nside': 1,
            'post_normalise': False,
            **transform_overrides
        }
        train_dict = {
            'patience': 20, 
            'learning_rate': 1e-5,
            'restore_from_previous': True,
            'weight_by_round': False,
            **training_overrides
        }
        mr_dict = {
            'simulation_budget': 50_000,
            'n_rounds': 15,
            'reference_theta': reference_theta,
            'dequantise_data': False,
            'initial_fraction': 0.5,
            'n_likelihood_samples': 25_000,
            **multiround_overrides
        }
        nle_dict = {
            'maf_stack_size': 15,
            'conditioner_n_layers': 4,
            'conditioner_n_neurons': 256, # don't drop these, hinders inference, 20250905_094425
            'decoder_n_layers': 3,
            'decoder_n_neurons': 64,
            'decoder_distribution': 'gaussian',
            **ssnle_overrides
        }
        train_config = TrainingConfig(**train_dict)
        mr_config = MultiRoundInfererConfig(**mr_dict)
        nle_config = SurjectiveNLEConfig.heirarchical(**nle_dict)
        transform_config = TransformConfig.haar_wavelet(**transform_dict)
        return cls(
            training_config=train_config,
            multiround_config=mr_config,
            ssnle_config=nle_config,
            transform_config=transform_config
        )
