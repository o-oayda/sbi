from abc import ABC
from dataclasses import dataclass, field, fields, make_dataclass, replace
from typing import Optional, Literal
from numpy.typing import NDArray
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.transforms import (
    DipoleBijectorWrapper,
    DipoleThetaTransform,
    InvertibleDataTransform,
    InvertibleThetaTransformJax,
    ZScore
)
from dipolesbi.tools.hadamard_transform import HadamardTransform, HadamardTransformJax
import jax
import numpy as np


def _make_override_class(base_cls):
    specs = []
    for f in fields(base_cls):
        if not f.init:
            continue
        specs.append((f.name, Optional[f.type], field(default=None)))

    override_cls = make_dataclass(
        f"{base_cls.__name__}Overrides",
        specs,
        frozen=False,
    )

    def to_dict(self):
        return {
            name: getattr(self, name) for name, *_ in specs
            if getattr(self, name) is not None
        }

    override_cls.to_dict = to_dict
    return override_cls

@dataclass
class ModelConfig(ABC):
    pass

@dataclass
class CatwiseConfig(ModelConfig):
    cat_w1_max: float
    cat_w12_min: float
    magnitude_error_dist: Literal['gaussian', 'students-t']
    use_float32: bool = False
    chunk_size: int = 25_000
    store_final_samples: bool = False
    use_common_extra_error: Optional[bool] = False
    model_identifier: Optional[str] = None
    downscale_nside: Optional[int] = None

    def __post_init__(self) -> None:
        if self.magnitude_error_dist not in ('gaussian', 'students-t'):
            raise ValueError(
                "Magnitude_error_dist must be 'gaussian' or 'students-t'."
            )

        if self.downscale_nside is not None:
            if self.downscale_nside <= 0:
                raise ValueError('downscale_nside must be a positive integer.')

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


TrainingConfigOverrides = _make_override_class(TrainingConfig)

@dataclass
class EmbeddingNetConfig:
    nside: Optional[int] = None
    out_channels_per_layer: Optional[list[int]] = None
    n_blocks: Optional[int] = None
    n_mlp_neurons: int = 64
    n_mlp_layers: int = 2
    output_dim: int = 32
    dropout_rate: float = 0.2

    def __post_init__(self) -> None:
        if self.n_blocks is None:
            assert self.nside is not None
            self.n_blocks = int(np.log2(self.nside))
        assert self.n_blocks > 0

EmbeddingNetConfigOverrides = _make_override_class(EmbeddingNetConfig)

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


NeuralFlowConfigOverrides = _make_override_class(NeuralFlowConfig)

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
    check_proposal_probs: bool = True
    write_results_to_disk: bool = True
    likelihood_chunk_size_gb: float = 0.25

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

@dataclass(frozen=True)
class DataTransformSpec:
    kind: Literal['blank', 'zscore', 'hadamard', 'hp_cnn'] = 'blank'
    embed_in_flow: bool = False
    zscore_method: Literal['batchwise', 'global'] = 'batchwise'
    hadamard_first_nside: Optional[int] = None
    hadamard_last_nside: Optional[int] = None
    hadamard_matrix_type: Optional[str] = None
    hadamard_normalise_details: Optional[bool] = None
    hadamard_n_chunks: Optional[int] = None
    embedding_config: Optional[EmbeddingNetConfig] = None

    @classmethod
    def blank(cls, *, embed_in_flow: bool = False) -> 'DataTransformSpec':
        return cls(kind='blank', embed_in_flow=embed_in_flow)

    @classmethod
    def zscore(
        cls,
        *,
        method: Literal['batchwise', 'global'] = 'batchwise',
        embed_in_flow: bool = False,
        embedding_config: Optional[EmbeddingNetConfig] = None,
    ) -> 'DataTransformSpec':
        return cls(
            kind='zscore',
            embed_in_flow=embed_in_flow,
            zscore_method=method,
            embedding_config=embedding_config,
        )

    @classmethod
    def hadamard(
        cls,
        *,
        first_nside: int = 16,
        last_nside: int = 1,
        matrix_type: str = 'hadamard',
        normalise_details: bool = True,
        n_chunks: int = 1,
        embed_in_flow: bool = True,
        embedding_config: Optional[EmbeddingNetConfig] = None,
    ) -> 'DataTransformSpec':
        return cls(
            kind='hadamard',
            embed_in_flow=embed_in_flow,
            hadamard_first_nside=first_nside,
            hadamard_last_nside=last_nside,
            hadamard_matrix_type=matrix_type,
            hadamard_normalise_details=normalise_details,
            hadamard_n_chunks=n_chunks,
            embedding_config=embedding_config,
        )

    @classmethod
    def hp_cnn_embed(
        cls,
        *,
        embedding_config: EmbeddingNetConfig,
        embed_in_flow: bool = False,
    ) -> 'DataTransformSpec':
        return cls(
            kind='hp_cnn', 
            embed_in_flow=embed_in_flow, 
            embedding_config=embedding_config
        )


def _build_data_transform(spec: DataTransformSpec) -> Optional[InvertibleDataTransform]:
    if spec.kind == 'blank' or spec.kind == 'hp_cnn':
        return None
    if spec.kind == 'zscore':
        return ZScore(method=spec.zscore_method)
    if spec.kind == 'hadamard':
        required = (
            spec.hadamard_first_nside,
            spec.hadamard_last_nside,
            spec.hadamard_matrix_type,
            spec.hadamard_normalise_details,
            spec.hadamard_n_chunks,
        )
        if any(value is None for value in required):
            raise ValueError('Incomplete hadamard data transform specification')
        params = dict(
            first_nside=spec.hadamard_first_nside,
            last_nside=spec.hadamard_last_nside,
            matrix_type=spec.hadamard_matrix_type,
            normalise_details=spec.hadamard_normalise_details,
            n_chunks=spec.hadamard_n_chunks,
        )
        if spec.embed_in_flow:
            jax.config.update('jax_enable_x64', True)
            return HadamardTransformJax(**params)  # type: ignore[arg-type]
        return HadamardTransform(**params)  # type: ignore[arg-type]
    raise ValueError(f"Unknown data transform spec kind '{spec.kind}'")


@dataclass
class DataTransformConfig:
    spec: DataTransformSpec
    _transform: Optional[InvertibleDataTransform] = field(default=None, init=False, repr=False)

    @classmethod
    def hp_cnn_embed(cls, nside, **embedd_net_overrides) -> 'DataTransformConfig':
        embedding_net_config = EmbeddingNetConfig(
            nside=nside,
            **embedd_net_overrides
        )
        spec = DataTransformSpec.hp_cnn_embed(embedding_config=embedding_net_config)
        return cls(spec)

    @classmethod
    def blank_transform(
        cls,
        embed_transform_in_flow: bool = False,
    ) -> 'DataTransformConfig':
        return cls(
            DataTransformSpec.blank(
                embed_in_flow=embed_transform_in_flow,
            )
        )

    @classmethod
    def zscore(
            cls,
            embed_transform_in_flow: bool = False,
            method: Literal['batchwise', 'global'] = 'batchwise',
            embedding_net_config: Optional[EmbeddingNetConfig] = None,
    ) -> 'DataTransformConfig':
        return cls(
            DataTransformSpec.zscore(
                method=method,
                embed_in_flow=embed_transform_in_flow,
                embedding_config=embedding_net_config,
            )
        )

    @classmethod
    def hadamard_wavelet(
            cls,
            first_nside: int = 16,
            last_nside: int = 1,
            matrix_type: str = 'hadamard',
            normalise_details: bool = True,
            n_chunks: int = 1,
            embed_transform_in_flow: bool = True,
            embedding_net_config: Optional[EmbeddingNetConfig] = None,
    ) -> 'DataTransformConfig':
        return cls(
            DataTransformSpec.hadamard(
                first_nside=first_nside,
                last_nside=last_nside,
                matrix_type=matrix_type,
                normalise_details=normalise_details,
                n_chunks=n_chunks,
                embed_in_flow=embed_transform_in_flow,
                embedding_config=embedding_net_config,
            )
        )

    @property
    def data_transform(self) -> Optional[InvertibleDataTransform]:
        if self.spec.kind in ('blank', 'hp_cnn'):
            return None
        if self._transform is None:
            self._transform = _build_data_transform(self.spec)
        return self._transform

    @property
    def embedding_net_config(self) -> Optional[EmbeddingNetConfig]:
        return self.spec.embedding_config

    @property
    def embed_transform_in_flow(self) -> bool:
        return self.spec.embed_in_flow

@dataclass(frozen=True)
class ThetaTransformSpec:
    kind: Literal['blank', 'dipole_cartesian', 'dipole_zscore', 'dipole_bijector'] = 'blank'
    embed_in_flow: bool = False
    wrap_longitude: Optional[bool] = None
    reflect_latitude: Optional[bool] = None

    @classmethod
    def blank(cls, *, embed_in_flow: bool = False) -> 'ThetaTransformSpec':
        return cls(kind='blank', embed_in_flow=embed_in_flow)

    @classmethod
    def dipole_cartesian(
        cls,
        *,
        embed_in_flow: bool = False,
        wrap_longitude: bool = True,
        reflect_latitude: bool = True,
    ) -> 'ThetaTransformSpec':
        return cls(
            kind='dipole_cartesian',
            embed_in_flow=embed_in_flow,
            wrap_longitude=wrap_longitude,
            reflect_latitude=reflect_latitude,
        )

    @classmethod
    def dipole_zscore(
        cls,
        *,
        embed_in_flow: bool = True,
        wrap_longitude: bool = True,
        reflect_latitude: bool = True,
    ) -> 'ThetaTransformSpec':
        return cls(
            kind='dipole_zscore',
            embed_in_flow=embed_in_flow,
            wrap_longitude=wrap_longitude,
            reflect_latitude=reflect_latitude,
        )

    @classmethod
    def dipole_bijector(
        cls,
        *,
        embed_in_flow: bool = True
    ) -> 'ThetaTransformSpec':
        return cls(
            kind='dipole_bijector',
            embed_in_flow=embed_in_flow
        )


def _build_theta_transform(
    spec: ThetaTransformSpec,
    prior: Optional[DipolePriorJax],
) -> Optional[InvertibleThetaTransformJax]:
    if spec.kind == 'blank':
        return None
    if prior is None:
        raise ValueError('Prior required to build theta transform')
    
    method = spec.kind.split('_')[1]
    if method in ['cartesian', 'zscore']:
        assert spec.wrap_longitude is not None
        assert spec.reflect_latitude is not None
        return DipoleThetaTransform(
            prior,
            method=method, # type: ignore
            wrap_longitude=spec.wrap_longitude,
            reflect_latitude=spec.reflect_latitude,
        )
    else:
        assert method == 'bijector', f'Method: {method}'
        return DipoleBijectorWrapper(prior=prior)


@dataclass
class ThetaTransformConfig:
    spec: ThetaTransformSpec
    prior: Optional[DipolePriorJax] = None
    _transform: Optional[InvertibleThetaTransformJax] = field(default=None, init=False, repr=False)

    @classmethod
    def blank_transform(
        cls,
        embed_transform_in_flow: bool = False
    ) -> 'ThetaTransformConfig':
        return cls(ThetaTransformSpec.blank(embed_in_flow=embed_transform_in_flow))

    @classmethod
    def dipole_cartesian_transform(
        cls,
        prior: DipolePriorJax,
        embed_transform_in_flow: bool = False,
        wrap_longitude: bool = True,
        reflect_latitude: bool = True
    ) -> 'ThetaTransformConfig':
        spec = ThetaTransformSpec.dipole_cartesian(
            embed_in_flow=embed_transform_in_flow,
            wrap_longitude=wrap_longitude,
            reflect_latitude=reflect_latitude,
        )
        return cls(spec=spec, prior=prior)

    @classmethod
    def dipole_zscore_transform(
        cls,
        prior: DipolePriorJax,
        embed_transform_in_flow: bool = True,
        wrap_longitude: bool = True,
        reflect_latitude: bool = True
    ) -> 'ThetaTransformConfig':
        spec = ThetaTransformSpec.dipole_zscore(
            embed_in_flow=embed_transform_in_flow,
            wrap_longitude=wrap_longitude,
            reflect_latitude=reflect_latitude,
        )
        return cls(spec=spec, prior=prior)

    @property
    def theta_transform(self) -> Optional[InvertibleThetaTransformJax]:
        if self.spec.kind == 'blank':
            return None
        if self._transform is None:
            self._transform = _build_theta_transform(self.spec, self.prior)
        return self._transform

    @property
    def embed_transform_in_flow(self) -> bool:
        return self.spec.embed_in_flow

@dataclass
class TransformConfig:
    data_transform_config: DataTransformConfig
    theta_transform_config: ThetaTransformConfig


def _apply_data_spec_overrides(
    spec: DataTransformSpec,
    overrides: Optional[dict]
) -> DataTransformSpec:
    if not overrides:
        return spec
    mapping = {
        'first_nside': 'hadamard_first_nside',
        'last_nside': 'hadamard_last_nside',
        'matrix_type': 'hadamard_matrix_type',
        'normalise_details': 'hadamard_normalise_details',
        'n_chunks': 'hadamard_n_chunks',
        'method': 'zscore_method',
        'embed_transform_in_flow': 'embed_in_flow',
    }
    updates = {}
    for key, value in overrides.items():
        target = mapping.get(key, key)
        if target not in DataTransformSpec.__dataclass_fields__:
            raise KeyError(f"Unknown data transform spec field '{key}'")
        updates[target] = value
    return replace(spec, **updates)


def _apply_theta_spec_overrides(
    spec: ThetaTransformSpec,
    overrides: Optional[dict]
) -> ThetaTransformSpec:
    if not overrides:
        return spec
    mapping = {
        'embed_transform_in_flow': 'embed_in_flow',
    }
    updates = {}
    for key, value in overrides.items():
        target = mapping.get(key, key)
        if target not in ThetaTransformSpec.__dataclass_fields__:
            raise KeyError(f"Unknown theta transform spec field '{key}'")
        updates[target] = value
    return replace(spec, **updates)


def _prepare_data_config(
    spec: DataTransformSpec,
    overrides: Optional[dict] = None,
) -> DataTransformConfig:
    updated_spec = _apply_data_spec_overrides(spec, overrides)
    return DataTransformConfig(updated_spec)


def _prepare_theta_config(
    spec: ThetaTransformSpec,
    prior: Optional[DipolePriorJax],
    overrides: Optional[dict] = None,
) -> ThetaTransformConfig:
    updated_spec = _apply_theta_spec_overrides(spec, overrides)
    if updated_spec.kind != 'blank' and prior is None:
        raise ValueError('Prior required for non-blank theta transform spec.')
    return ThetaTransformConfig(
        spec=updated_spec,
        prior=prior if updated_spec.kind != 'blank' else None,
    )


def _sync_flow_embed_flag(flow_cfg: NeuralFlowConfig, transforms: TransformConfig) -> None:
    if flow_cfg.mode == 'NLE':
        flow_cfg.embed_target_transform_in_flow = transforms.data_transform_config.embed_transform_in_flow
    elif flow_cfg.mode == 'NPE':
        flow_cfg.embed_target_transform_in_flow = transforms.theta_transform_config.embed_transform_in_flow
    else:
        flow_cfg.embed_target_transform_in_flow = False


@dataclass
class Scenario:
    training: TrainingConfig
    multiround: MultiRoundInfererConfig
    flow: NeuralFlowConfig
    transforms: TransformConfig

    @classmethod
    def blank(
        cls,
        reference_theta: dict[str, NDArray],
        *,
        theta_prior: Optional[DipolePriorJax] = None,
        training_overrides: Optional[dict] = None,
        multiround_overrides: Optional[dict] = None,
        flow_overrides: Optional[dict] = None,
        data_spec: Optional[DataTransformSpec] = None,
        theta_spec: Optional[ThetaTransformSpec] = None,
        data_spec_overrides: Optional[dict] = None,
        theta_spec_overrides: Optional[dict] = None,
    ) -> 'Scenario':
        train_cfg = TrainingConfig(**(training_overrides or {}))
        mr_defaults = {
            'simulation_budget': 10_000,
            'n_rounds': 15,
            'reference_theta': reference_theta,
        }
        mr_defaults.update(multiround_overrides or {})
        mr_cfg = MultiRoundInfererConfig(**mr_defaults)

        flow_defaults = {
            'mode': 'NLE',
            'architecture': 3 * ['MAF'],
        }
        flow_defaults.update(flow_overrides or {})
        flow_cfg = NeuralFlowConfig(**flow_defaults)

        base_data_spec = data_spec or DataTransformSpec.blank(embed_in_flow=False)
        data_cfg = _prepare_data_config(base_data_spec, data_spec_overrides)

        base_theta_spec = theta_spec or ThetaTransformSpec.blank(embed_in_flow=False)
        theta_cfg = _prepare_theta_config(
            base_theta_spec, theta_prior, theta_spec_overrides
        )

        transforms = TransformConfig(
            data_transform_config=data_cfg,
            theta_transform_config=theta_cfg,
        )

        _sync_flow_embed_flag(flow_cfg, transforms)

        return cls(train_cfg, mr_cfg, flow_cfg, transforms)

    # good for nside <= 16
    @classmethod
    def anynside_nle(
        cls,
        nside: int,
        theta_prior: DipolePriorJax,
        *,
        reference_theta: Optional[dict[str, NDArray]] = None,
        training_overrides: Optional[dict] = None,
        multiround_overrides: Optional[dict] = None,
        flow_overrides: Optional[dict] = None,
        data_spec: Optional[DataTransformSpec] = None,
        theta_spec: Optional[ThetaTransformSpec] = None,
        data_spec_overrides: Optional[dict] = None,
        theta_spec_overrides: Optional[dict] = None,
    ) -> 'Scenario':
        train_defaults = {
            'patience': 20,
            'learning_rate': 5e-5,
            'restore_from_previous': False,
            'weight_by_round': False,
        }
        train_defaults.update(training_overrides or {})
        train_cfg = TrainingConfig(**train_defaults)

        mr_defaults = {
            'simulation_budget': 50_000,
            'n_rounds': 15,
            'reference_theta': reference_theta,
            'dequantise_data': False,
            'initial_fraction': 0.,
            'n_likelihood_samples': 20_000,
        }
        mr_defaults.update(multiround_overrides or {})
        mr_cfg = MultiRoundInfererConfig(**mr_defaults)

        flow_defaults = {
            'mode': 'NLE',
            'architecture': ['healpix_funnel'],
            'funnel_one_and_done': False,
            'funnel_maf_extension': 2,
            'conditioner_n_layers': 4,
            'conditioner_n_neurons': 256,
            'decoder_n_layers': 3,
            'decoder_n_neurons': 64,
            'decoder_distribution': 'gaussian',
        }
        flow_defaults.update(flow_overrides or {})
        flow_cfg = NeuralFlowConfig(**flow_defaults)

        base_data_spec = data_spec or DataTransformSpec.hadamard(
            first_nside=nside,
            last_nside=1,
            matrix_type='hadamard',
            normalise_details=True,
            n_chunks=1,
            embed_in_flow=False,
        )
        data_cfg = _prepare_data_config(base_data_spec, data_spec_overrides)

        base_theta_spec = theta_spec or ThetaTransformSpec.dipole_cartesian(
            embed_in_flow=False
        )
        theta_cfg = _prepare_theta_config(
            base_theta_spec, theta_prior, theta_spec_overrides
        )

        transforms = TransformConfig(
            data_transform_config=data_cfg,
            theta_transform_config=theta_cfg,
        )

        _sync_flow_embed_flag(flow_cfg, transforms)

        return cls(train_cfg, mr_cfg, flow_cfg, transforms)

    @classmethod
    def anynside_npe(
        cls,
        nside: int,
        theta_prior: DipolePriorJax,
        *,
        reference_theta: Optional[dict[str, NDArray]] = None,
        training_overrides: Optional[dict] = None,
        multiround_overrides: Optional[dict] = None,
        flow_overrides: Optional[dict] = None,
        data_spec: Optional[DataTransformSpec] = None,
        theta_spec: Optional[ThetaTransformSpec] = None,
        data_spec_overrides: Optional[dict] = None,
        theta_spec_overrides: Optional[dict] = None,
    ) -> 'Scenario':
        # NPE needs a generally higher lr than NLE
        train_defaults = {
            'patience': 30,
            'min_lr_ratio': 1.,
            'learning_rate': 5e-4,
            'restore_from_previous': False
        }
        train_defaults.update(training_overrides or {})
        train_cfg = TrainingConfig(**train_defaults)

        mr_defaults = {
            'simulation_budget': 50_000,
            'n_rounds': 3,
            'reference_theta': reference_theta,
            'initial_fraction': 0.,
            'n_likelihood_samples': 25_000,
        }
        mr_defaults.update(multiround_overrides or {})
        mr_cfg = MultiRoundInfererConfig(**mr_defaults)

        flow_defaults = {
            'mode': 'NPE',
            'architecture': 5 * ['MAF'],
            'conditioner_n_layers': 2,
            'conditioner_n_neurons': 64,
        }
        flow_defaults.update(flow_overrides or {})
        flow_cfg = NeuralFlowConfig(**flow_defaults)

        embedding_cfg = EmbeddingNetConfig(
            nside=nside,
            out_channels_per_layer=[2, 4, 8, 16], # don't bump thse too high
            dropout_rate=0.2,
            n_mlp_neurons=128,
            n_blocks=int(np.log2(nside) - 2)
        )
        base_data_spec = data_spec or DataTransformSpec.zscore(
            method='global', # use global for npe
            embedding_config=embedding_cfg,
        )
        data_cfg = _prepare_data_config(base_data_spec, data_spec_overrides)

        base_theta_spec = theta_spec or ThetaTransformSpec.dipole_bijector(embed_in_flow=True)
        theta_cfg = _prepare_theta_config(base_theta_spec, theta_prior, theta_spec_overrides)

        transforms = TransformConfig(
            data_transform_config=data_cfg,
            theta_transform_config=theta_cfg,
        )

        _sync_flow_embed_flag(flow_cfg, transforms)

        return cls(train_cfg, mr_cfg, flow_cfg, transforms)

    @classmethod
    def nside32(
        cls,
        reference_theta: dict[str, NDArray],
        theta_prior: DipolePriorJax,
        *,
        training_overrides: Optional[dict] = None,
        multiround_overrides: Optional[dict] = None,
        flow_overrides: Optional[dict] = None,
        data_spec: Optional[DataTransformSpec] = None,
        theta_spec: Optional[ThetaTransformSpec] = None,
        data_spec_overrides: Optional[dict] = None,
        theta_spec_overrides: Optional[dict] = None,
    ) -> 'Scenario':
        train_defaults = {
            'patience': 20,
            'learning_rate': 1e-5,
            'restore_from_previous': True,
            'weight_by_round': False,
        }
        train_defaults.update(training_overrides or {})
        train_cfg = TrainingConfig(**train_defaults)

        mr_defaults = {
            'simulation_budget': 50_000,
            'n_rounds': 15,
            'reference_theta': reference_theta,
            'dequantise_data': False,
            'initial_fraction': 0.5,
            'n_likelihood_samples': 25_000,
        }
        mr_defaults.update(multiround_overrides or {})
        mr_cfg = MultiRoundInfererConfig(**mr_defaults)

        flow_defaults = {
            'mode': 'NLE',
            'architecture': ['healpix_funnel'] + 15 * ['MAF'],
            'funnel_one_and_done': False,
            'funnel_maf_extension': 0,
            'conditioner_n_layers': 4,
            'conditioner_n_neurons': 256,
            'decoder_n_layers': 3,
            'decoder_n_neurons': 64,
            'decoder_distribution': 'gaussian',
        }
        flow_defaults.update(flow_overrides or {})
        flow_cfg = NeuralFlowConfig(**flow_defaults)

        base_data_spec = data_spec or DataTransformSpec.hadamard(
            first_nside=32,
            last_nside=1,
            matrix_type='hadamard',
            normalise_details=True,
            n_chunks=1,
            embed_in_flow=False,
        )
        data_cfg = _prepare_data_config(base_data_spec, data_spec_overrides)

        base_theta_spec = theta_spec or ThetaTransformSpec.dipole_cartesian(
            embed_in_flow=False
        )
        theta_cfg = _prepare_theta_config(
            base_theta_spec, theta_prior, theta_spec_overrides
        )

        transforms = TransformConfig(
            data_transform_config=data_cfg,
            theta_transform_config=theta_cfg,
        )

        _sync_flow_embed_flag(flow_cfg, transforms)

        return cls(train_cfg, mr_cfg, flow_cfg, transforms)
