from dataclasses import asdict, dataclass, field
from typing import Optional, Literal
from numpy.typing import NDArray
from dipolesbi.tools.transforms import DipoleThetaTransform,  InvertibleDataTransform, InvertibleThetaTransformJax, ZScore
from dipolesbi.tools.hadamard_transform import HadamardTransform, HadamardTransformJax
import jax
from dipolesbi.tools.utils import PytreeAdapter


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
class NeuralFlowConfig:
    '''
    Configuration for the NLE.
    '''
    mode: Literal['NLE', 'NPE']
    # layer types
    architecture: list[
        Literal['MAF'] 
      | Literal['healpix_funnel'] 
      | Literal['surjective_MAF']
    ]
    surjective_layer_type: Literal[
        'affine_MAF', 'rational_quadratic_MAF'
    ] = 'affine_MAF'
    conditioner: Literal['made', 'mlp', 'transformer'] = 'made'

    # layer hyperparams
    decoder_distribution: Literal['gaussian', 'poisson', 'nb', 'students_t'] = 'gaussian'
    decoder_n_neurons: Optional[int] = None
    decoder_n_layers: Optional[int] = None
    conditioner_n_neurons: int = 128
    conditioner_n_layers: int = 3

    # healfunnel options
    funnel_one_and_done: Optional[bool] = None
    funnel_maf_extension: Optional[int] = None

    # generic surjective MAF options
    data_reduction_factor: Optional[float] = None

    conditioner_kwargs: dict = field(default_factory=dict)
    embed_target_transform_in_flow: bool = False

    def __post_init__(self):
        if self.surjective_layer_type in ['affine_MAF', 'rational_quadratic_MAF']:
            assert self.conditioner == 'made', (
                f'Conditioner must be MADE when using {self.surjective_layer_type}.'
            )

        if 'healpix_funnel' in self.architecture:
            assert self.funnel_one_and_done is not None
            assert self.funnel_maf_extension is not None
            assert self.funnel_maf_extension % 2 == 0

            # assert even number of mafs before healpix funnel
            # each maf has a inverse order perm, so we need the data in the OG order
            idx = self.architecture.index('healpix_funnel')
            maf_count = sum(1 for x in self.architecture[:idx] if x == 'MAF')
            assert maf_count % 2 == 0, (
                "Number of 'MAF' before 'healpix_funnel' must be even"
            )

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
    n_posterior_samples: int = 10_000 # for NPE after training

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
class DataTransformConfig:
    data_transform: Optional[InvertibleDataTransform] = None
    first_nside: Optional[int] = None
    last_nside: Optional[int] = None
    matrix_type: Optional[str] = None
    normalise_details: Optional[bool] = None
    n_chunks: Optional[int] = None
    embed_transform_in_flow: Optional[bool] = None

    @classmethod
    def blank_transform(cls, embed_transform_in_flow: bool = False) -> 'DataTransformConfig':
        config = cls(embed_transform_in_flow=embed_transform_in_flow)
        config.data_transform = None
        return config

    @classmethod
    def zscore(
            cls,
            embed_transform_in_flow: bool = False, 
            method: Literal['batchwise', 'global'] = 'batchwise'
    ) -> 'DataTransformConfig':
        data_transform = ZScore(method=method)
        config = cls(embed_transform_in_flow=embed_transform_in_flow)
        config.data_transform = data_transform
        return config

    @classmethod
    def hadamard_wavelet(
            cls,
            first_nside: int = 16,
            last_nside: int = 1,
            matrix_type: str = 'hadamard',
            normalise_details: bool = True,
            n_chunks: int = 1,
            embed_transform_in_flow: bool = True,
    ) -> 'DataTransformConfig':
        config = cls(
            first_nside=first_nside,
            last_nside=last_nside,
            matrix_type=matrix_type,
            normalise_details=normalise_details,
            n_chunks=n_chunks,
            embed_transform_in_flow=embed_transform_in_flow
        )
        config_dict = asdict(config)

        # this is kind of bullshit
        del config_dict['embed_transform_in_flow']
        del config_dict['data_transform']

        # if embedding, we need float64 computation; otherwise, use numpy
        if config.embed_transform_in_flow:
            jax.config.update('jax_enable_x64', True)
            data_transform_instance = HadamardTransformJax(**config_dict)
        else:
            data_transform_instance = HadamardTransform(**config_dict)

        config.data_transform = data_transform_instance
        return config

@dataclass
class ThetaTransformConfig:
    theta_transform: Optional[InvertibleThetaTransformJax] = None
    embed_transform_in_flow: Optional[bool] = None
    dipole_theta_method: Optional[Literal['cartesian', 'zscore']] = None
    wrap_longitude: Optional[bool] = True
    reflect_latitude: Optional[bool] = True

    @classmethod
    def blank_transform(
        cls,
        embed_transform_in_flow: bool = False
    ) -> 'ThetaTransformConfig':
        theta_transform = None
        return cls(
            theta_transform=theta_transform,
            embed_transform_in_flow=embed_transform_in_flow
        )
    @classmethod
    def dipole_cartesian_transform(
        cls,
        pytree_adapter: PytreeAdapter,
        embed_transform_in_flow: bool = False,
        wrap_longitude: bool = True,
        reflect_latitude: bool = True
    ) -> 'ThetaTransformConfig':
        theta_transform = DipoleThetaTransform(
            pytree_adapter,
            method='cartesian',
            wrap_longitude=wrap_longitude,
            reflect_latitude=reflect_latitude
        )
        return cls(
            theta_transform=theta_transform,
            embed_transform_in_flow=embed_transform_in_flow,
            dipole_theta_method='cartesian'
        )

    @classmethod
    def dipole_zscore_transform(
        cls,
        pytree_adapter: PytreeAdapter,
        embed_transform_in_flow: bool = True,
        wrap_longitude: bool = True,
        reflect_latitude: bool = True
    ) -> 'ThetaTransformConfig':
        theta_transform = DipoleThetaTransform(
            pytree_adapter, 
            method='zscore',
            wrap_longitude=wrap_longitude,
            reflect_latitude=reflect_latitude
        )
        return cls(
            theta_transform=theta_transform,
            embed_transform_in_flow=embed_transform_in_flow,
            dipole_theta_method='zscore'
        )

# TODO: refactor out a data_transform and theta_transform subconfig
@dataclass
class TransformConfig:
    data_transform_config: DataTransformConfig
    theta_transform_config: ThetaTransformConfig

    @classmethod
    def blank_transform(cls) -> 'TransformConfig':
        data_transform_config = DataTransformConfig.blank_transform()
        theta_transform_config = ThetaTransformConfig.blank_transform()
        return cls(
            data_transform_config=data_transform_config,
            theta_transform_config=theta_transform_config
        )

    @classmethod
    def hadamard_for_nle(
        cls,
        adapter: PytreeAdapter,
        data_transform_overrides: dict = {},
        theta_transform_overrides: dict = {},
    ) -> 'TransformConfig':
        data_transform_config_dict = {
            'first_nside': 16,
            'last_nside': 1,
            'matrix_type': 'hadamard',
            'normalise_details': True,
            'n_chunks': 1,
            'embed_transform_in_flow': False,
            **data_transform_overrides
        }
        theta_transform_config_dict = {
            'pytree_adapter': adapter,
            'embed_transform_in_flow': None,
            **theta_transform_overrides
        }
        data_transform_config = DataTransformConfig.hadamard_wavelet(
            **data_transform_config_dict
        )
        theta_transform_config = ThetaTransformConfig.dipole_cartesian_transform(
            **theta_transform_config_dict
        )
        return cls(
            data_transform_config=data_transform_config,
            theta_transform_config=theta_transform_config
        )

    @classmethod
    def raw_data_for_npe(
            cls,
            adapter: PytreeAdapter,
            data_transform_overrides: dict = {},
            theta_transform_overrides: dict = {}
    ) -> 'TransformConfig':
        data_transform_config_dict = {
            'embed_transform_in_flow': None,
            **data_transform_overrides
        }
        data_transform_config = DataTransformConfig.zscore(**data_transform_config_dict)
        # data_transform_config = DataTransformConfig.hadamard_wavelet(
            # embed_transform_in_flow=False
        # )
        theta_transform_config_dict = {
            'pytree_adapter': adapter,
            'embed_transform_in_flow': True,
            **theta_transform_overrides
        }
        theta_transform_config = ThetaTransformConfig.dipole_zscore_transform(
            **theta_transform_config_dict
        )
        return cls(
            data_transform_config=data_transform_config,
            theta_transform_config=theta_transform_config
        )
        
@dataclass
class ConfigOfConfigs:
    '''
    Meta configuration class for configs which, from experimenting, work well in
    particular settings. Access each setting by calling the relevant class method.
    '''
    training_config: TrainingConfig
    multiround_config: MultiRoundInfererConfig
    ssnle_config: NeuralFlowConfig
    transform_config: TransformConfig

    # no idea where to put this logic for now
    def __post_init__(self) -> None:
        if (
            (self.ssnle_config.mode == 'NLE')
            and self.transform_config.data_transform_config.embed_transform_in_flow
        ):
            self.ssnle_config.embed_target_transform_in_flow = True
        elif (
            (self.ssnle_config.mode == 'NPE')
            and self.transform_config.theta_transform_config.embed_transform_in_flow
            ):
            self.ssnle_config.embed_target_transform_in_flow = True
        else:
            self.embed_target_transform_in_flow = False

    @classmethod
    def blank(
            cls,
            reference_theta: dict[str, NDArray],
            training_overrides: dict = {},
            multiround_overrides: dict = {},
            ssnle_overrides: dict = {}
    ) -> 'ConfigOfConfigs':
        mr_config_dict = {
            'simulation_budget': 10_000,
            'n_rounds': 15,
            'reference_theta': reference_theta,
            **multiround_overrides
        }
        nle_config_dict = {
            'mode': 'NLE',
            'architecture': 3 * ['MAF'],
            **ssnle_overrides
        }

        transform_config = TransformConfig.blank_transform()
        train_config = TrainingConfig(**training_overrides)
        mr_config = MultiRoundInfererConfig(**mr_config_dict)
        nle_config = NeuralFlowConfig(**nle_config_dict)

        return cls(
            training_config=train_config,
            multiround_config=mr_config,
            ssnle_config=nle_config,
            transform_config=transform_config
        )

    # low learning rate high nside?
    # ok not weighting by round helps in keeping posterior narrow
    @classmethod
    def nside16_nle(
            cls,
            reference_theta: dict[str, NDArray],
            theta_adapter: PytreeAdapter,
            training_overrides: dict = {},
            multiround_overrides: dict = {},
            nflow_overrides: dict = {},
            data_transform_overrides: dict = {},
            theta_transform_overrides: dict = {}
    ) -> 'ConfigOfConfigs':
        data_transform_dict = {
            'first_nside': 16,
            'last_nside': 1,
            'matrix_type': 'hadamard',
            'normalise_details': True,
            'n_chunks': 1,
            **data_transform_overrides
        }
        theta_transform_dict = theta_transform_overrides
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
        nflow_dict = {
            'mode': 'NLE',
            'architecture': ['healpix_funnel'] + 15 * ['MAF'],
            'funnel_one_and_done': False,
            'funnel_maf_extension': 0,
            'conditioner_n_layers': 4,
            'conditioner_n_neurons': 256,
            'decoder_n_layers': 3,
            'decoder_n_neurons': 64, # keep decoder_n_neurons ~ 64 for nside=16?
            'decoder_distribution': 'gaussian',
            **nflow_overrides
        }

        train_config = TrainingConfig(**train_dict)
        mr_config = MultiRoundInfererConfig(**mr_dict)
        nle_config = NeuralFlowConfig(**nflow_dict)
        transform_config = TransformConfig.hadamard_for_nle(
            adapter=theta_adapter,
            data_transform_overrides=data_transform_dict,
            theta_transform_overrides=theta_transform_dict
        )

        return cls(
            training_config=train_config,
            multiround_config=mr_config,
            ssnle_config=nle_config,
            transform_config=transform_config
        )

    @classmethod
    def nside16_npe(
            cls,
            reference_theta: dict[str, NDArray],
            theta_adapter: PytreeAdapter,
            training_overrides: dict = {},
            multiround_overrides: dict = {},
            nflow_overrides: dict = {},
            data_transform_overrides: dict = {},
            theta_transform_overrides: dict = {}
    ) -> 'ConfigOfConfigs':
        data_transform_dict = {
            **data_transform_overrides
        }
        theta_transform_dict = {
            'embed_transform_in_flow': True,
            **theta_transform_overrides
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
            'mode': 'NPE',
            'architecture': 8 * ['MAF'],
            'funnel_one_and_done': False,
            'funnel_maf_extension': 0,
            'conditioner_n_layers': 4,
            'conditioner_n_neurons': 256,
            **nflow_overrides
        }

        train_config = TrainingConfig(**train_dict)
        mr_config = MultiRoundInfererConfig(**mr_dict)
        nle_config = NeuralFlowConfig(**nle_dict)
        transform_config = TransformConfig.raw_data_for_npe(
            adapter=theta_adapter,
            data_transform_overrides=data_transform_dict,
            theta_transform_overrides=theta_transform_dict
        )

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
            theta_adapter: PytreeAdapter,
            training_overrides: dict = {},
            multiround_overrides: dict = {},
            ssnle_overrides: dict = {},
            data_transform_overrides: dict = {},
            theta_transform_overrides: dict = {}
    ) -> 'ConfigOfConfigs':
        data_transform_dict = {
            'first_nside': 32,
            'last_nside': 1,
            **data_transform_overrides
        }
        theta_transform_dict = {
            'embed_transform_in_flow': False,
            **theta_transform_overrides
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
            'mode': 'NLE',
            'architecture': ['healpix_funnel'] + 15 * ['MAF'],
            'funnel_one_and_done': False,
            'funnel_maf_extension': 0,
            'conditioner_n_layers': 4,
            'conditioner_n_neurons': 256, # don't drop these, hinders inference, 20250905_094425
            'decoder_n_layers': 3,
            'decoder_n_neurons': 64,
            'decoder_distribution': 'gaussian',
            **ssnle_overrides
        }

        train_config = TrainingConfig(**train_dict)
        mr_config = MultiRoundInfererConfig(**mr_dict)
        nle_config = NeuralFlowConfig(**nle_dict)
        transform_config = TransformConfig.hadamard_for_nle(
            adapter=theta_adapter,
            data_transform_overrides=data_transform_dict,
            theta_transform_overrides=theta_transform_dict
        )

        return cls(
            training_config=train_config,
            multiround_config=mr_config,
            ssnle_config=nle_config,
            transform_config=transform_config
        )
