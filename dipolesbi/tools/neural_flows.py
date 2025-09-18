from typing import Callable, Literal, Optional
import distrax
import haiku as hk
from haiku._src.transform import Transformed
from haiku._src.typing import PRNGKey
from jax import numpy as jnp
from numpy.typing import NDArray
from surjectors import (
    Chain,
    MaskedAutoregressive,
    Permutation,
    RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel,
    TransformedDistribution,
)
from surjectors.nn import MADE, make_mlp, make_transformer
from surjectors.util import unstack, named_dataset
from surjectors import AffineMaskedAutoregressiveInferenceFunnel
import optax
import jax
import numpy as np
import sys
import matplotlib.pyplot as plt
import logging
from abc import ABC, abstractmethod
from dipolesbi.tools.configs import NeuralFlowConfig, TrainingConfig
from dipolesbi.tools.distributions import IndependentWrapper, NegBinomDist, PoissonDist, StudentT
from dipolesbi.tools.dataloader import as_batch_iterator_cpu2gpu, named_dataset_idx
from dipolesbi.tools.np_rngkey import npkey_sequence_from_hk
from dipolesbi.tools.transforms import HaarWaveletTransform, InvertibleDataTransform
from dipolesbi.tools.ui import MultiRoundInfererUI


def make_mlp_with_dropout(sizes, activation=jax.nn.silu, dropout_rate=0.0):
    def net(x, *, training: bool):
        for h in sizes[:-1]:
            x = hk.Linear(h)(x)
            x = activation(x)
            if dropout_rate and dropout_rate > 0.0:
                x = hk.dropout(hk.next_rng_key(), dropout_rate, x) if training else x
        return hk.Linear(sizes[-1])(x)
    return net  # callable: net(x, training=bool)

class AbstractNeuralFlow(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.model = self.get_flow()
        self.best_params = None

    @abstractmethod
    def _flow(self, method: str, **kwargs) -> Transformed:
        pass

    def get_flow(self) -> Transformed:
        return hk.transform(self._flow)

    def _make_lr_schedule(
            self,
            warmup_steps: int,
            total_steps: int,
            base_lr: float,
            min_lr_ratio: float = 0.1
    ):
        assert warmup_steps < total_steps
        warmup = optax.linear_schedule(
            init_value=0., 
            end_value=base_lr,
            transition_steps=warmup_steps
        )
        cosine = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=min_lr_ratio  # final LR = alpha * base_lr
        )
        return optax.join_schedules([warmup, cosine], [warmup_steps])

    def _make_optimiser(
            self,
            lr_schedule,
            clip_norm: float = 1.,
            weight_decay: float = 0.
    ):
        return optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
        )

    def plot_loss_curve(self, show: bool = True, save_path: Optional[str] = None) -> None:
        _, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:orange'
        ax2.set_ylabel('Val Loss', color=color)
        ax2.plot(self.val_losses, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        if show:
            plt.show()

    def _make_loss_fn(
            self, 
            mode: Literal['NPE'] | Literal['NLE'],
            round_weights: Optional[jnp.ndarray] = None,
    ):
        def _nle_loss_fn(params, **batch):
            nlp = self._get_nlp(params, **batch)
            if round_weights is not None:
                w = round_weights[batch['round_id']]
                w = w * (w.size / (jnp.sum(w) + 1e-12))
                return jnp.mean(w * nlp)
            else:
                return jnp.mean(nlp)

        def _npe_loss_fn(params, **batch):
            pass

        if mode == 'NPE':
            return _npe_loss_fn
        elif mode == 'NLE':
            return _nle_loss_fn
        else:
            raise Exception(f'Mode {mode} not recognised.')

    def _get_nlp(self, params, **batch):
        lp = self.model.apply(
            params, None, method='log_prob', **self._model_kwargs(**batch)
        )
        return -lp

    def _make_round_weights(self, n_rounds: int, alpha: float = 1.):
        t = jnp.arange(n_rounds)
        w = jnp.exp(alpha * (t - (n_rounds - 1)))
        return w / jnp.mean(w)

    def _model_kwargs(self, **batch):
        allowed = {"y", "x"}
        return {k: v for k, v in batch.items() if k in allowed}

    def train(
            self,
            rng_seq: hk.PRNGSequence, 
            training_data: named_dataset_idx, 
            validation_data: named_dataset,
            config: TrainingConfig = TrainingConfig(),
            ui: Optional[MultiRoundInfererUI] = None
    ) -> tuple[NDArray, NDArray]:
        assert self.model is not None
        self.trn_config = config
        np_sequence = npkey_sequence_from_hk(rng_seq)

        if ui:
            print_func = ui.log
        else:
            print_func = print

        train_iter = as_batch_iterator_cpu2gpu(
            rng_key=next(np_sequence), 
            data=training_data, 
            batch_size=self.trn_config.batch_size,
            shuffle=self.trn_config.shuffle_train
        )
        val_iter = as_batch_iterator_cpu2gpu(
            rng_key=next(np_sequence), 
            data=validation_data,
            batch_size=self.trn_config.batch_size,
            shuffle=self.trn_config.shuffle_val
        )
        n_batches = train_iter.num_batches
        # assert n_batches == val_iter.num_batches
        steps_per_epoch = n_batches

        warmup_steps = max(
            10, int(self.trn_config.warmup_epochs * steps_per_epoch)
        )
        total_steps = max(
            steps_per_epoch * self.trn_config.max_n_iter, warmup_steps + 1
        )
        
        cur_sim_round = max(training_data.round_id) + 1
        if self.trn_config.weight_by_round:
            round_weights = self._make_round_weights(
                cur_sim_round, 
                self.trn_config.alpha_weight
            )
        else:
            round_weights = None
        print_func(f'Round weights: {round_weights}')

        if (self.trn_config.restore_from_previous) and (self.best_params is not None):
            print_func('Initialising using previously-inferred params...')
            params = self.best_params
        else:
            print_func('No previous state to initialise from. Starting fresh...')
            params = self.model.init(
                next(rng_seq), 
                method='log_prob', 
                **self._model_kwargs(**train_iter(0))
            )

        lr_schedule = self._make_lr_schedule(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=self.trn_config.learning_rate,
            min_lr_ratio=self.trn_config.min_lr_ratio
        )
        optimiser = self._make_optimiser(
            lr_schedule,
            clip_norm=self.trn_config.clip_norm,
            weight_decay=self.trn_config.weight_decay
        )
        # optimiser = optax.adam(self.trn_config.learning_rate, b2=self.trn_config.adam_b2)
        state = optimiser.init(params)
        loss_fn = self._make_loss_fn(round_weights)
        
        @jax.jit
        def step(params, state, **batch):
            # def loss_fn(params):
            #     lp = self.model.apply(params, None, method="log_prob", **batch)
            #     return -jnp.sum(lp)
            loss, grads = jax.value_and_grad(loss_fn)(params, **batch)
            updates, new_state = optimiser.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_state
        
        @jax.jit
        def val_step(params, **batch):
            return jnp.mean(self._get_nlp(params, **batch))

        losses = np.nan * np.zeros(self.trn_config.max_n_iter)
        val_losses = np.nan * np.zeros(self.trn_config.max_n_iter)
        best_params = params
        best_val = np.inf
        wait = 0

        for i in range(self.trn_config.max_n_iter):
            train_loss = 0.0
            for j in range(train_iter.num_batches):
                batch = train_iter(j)
                batch_loss, params, state = step(params, state, **batch)
                train_loss += float(batch_loss)
            train_loss /= max(1, train_iter.num_batches)
            
            val_loss = 0.0
            for j in range(val_iter.num_batches):
                batch = val_iter(j)
                val_loss += float(val_step(params, **batch))
            val_loss /= max(1, val_iter.num_batches)

            if ui:
                ui.set_subtitle(
                    f"Iter: {i} | "
                    f"Train NLL: {train_loss:.4f} | "
                    f"Val NLL: {val_loss:.4f} | "
                    f"Stop at {self.trn_config.patience} ({wait})"
                )
            else:
                sys.stdout.write(
                    f"\rIter: {i} | "
                    f"Train NLL: {train_loss:.4f} | "
                    f"Val NLL: {val_loss:.4f} | "
                    f"Stop at {self.trn_config.patience} ({wait})"
                )
                sys.stdout.flush()

            losses[i] = train_loss
            val_losses[i] = val_loss

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_params = params
                wait = 0
            else:
                wait += 1
                if i >= self.trn_config.min_n_iter and wait >= self.trn_config.patience:
                    print_func(
                        f"Early stopping at iteration {i} "
                        f"(best val NLL={best_val:.4f})"
                    )
                    break

        self.best_params = best_params
        self.losses = losses
        self.val_losses = val_losses

        return losses, val_losses

    def sample_likelihood_func(
            self, 
            rng_key: PRNGKey, 
            theta0: jnp.ndarray,
            **kwargs
    ) -> jnp.ndarray:
        samples = self.model.apply(
            self.best_params,
            rng_key,
            method='sample',
            x=theta0,
            **kwargs
        )
        return samples

    def evaluate_lnlike(
        self,
        theta0: jnp.ndarray,
        x0: jnp.ndarray
    ) -> jnp.ndarray:
        logprob = self.model.apply(
            params=self.best_params,
            rng=None, # don't pass key
            method='log_prob',
            x=theta0,
            y=x0
        )
        return logprob

class MAFNeuralLikelihood(AbstractNeuralFlow):
    def __init__(
            self, 
            data_ndim: int, 
            n_layers: int = 3,
            conditioner_neurons_per_layer: int = 64,
            conditioner_n_hidden_layers: int = 2
    ) -> None:
        self.n_layers = n_layers
        self.data_ndim = data_ndim
        self.conditioner_neurons_per_layer = conditioner_neurons_per_layer
        self.conditioner_n_hidden_layers = conditioner_n_hidden_layers
        self.model = self.get_flow()

    def _bijector_fn(self, params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(self, method: str, **kwargs):
        layers = []
        order = jnp.arange(self.data_ndim)
        n_neurons = self.conditioner_neurons_per_layer
        n_layers = self.conditioner_n_hidden_layers

        for _ in range(self.n_layers):
            layer = MaskedAutoregressive(
                bijector_fn=self._bijector_fn,
                conditioner=MADE(
                    input_size=self.data_ndim,
                    hidden_layer_sizes=n_layers * [n_neurons],
                    n_params=2, # mean and scale
                    w_init=hk.initializers.TruncatedNormal(0.01),
                    b_init=jnp.zeros,
                )
            )
            order = order[::-1]
            layers.append(layer)
            layers.append(Permutation(order, 1))

        layers = layers[:-1]
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(self.data_ndim), jnp.ones(self.data_ndim)),
            reinterpreted_batch_ndims=1
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method, **kwargs)


class NeuralFlow(AbstractNeuralFlow):
    def __init__(
            self,
            target_ndim: int, 
            config: NeuralFlowConfig,
            data_transform: Optional[InvertibleDataTransform] = None,
    ) -> None:
        '''
        If using a heirarchical flow, pass an invertible data transform so
        the detail blocks can be accessed.
        '''
        super().__init__()
        self.target_ndim = target_ndim
        self.nflow_config = config
        self.data_transform = data_transform
        self.mode = self.nflow_config.mode # NLE or NPE
        
        if isinstance(self.data_transform, HaarWaveletTransform):
            self.blocks = self.data_transform.blocks
        else:
            self.blocks = None

        self.model = self.get_flow()

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for i, layer in enumerate(getattr(self, "layers", [])):
            # Name of the layer
            name = layer.__class__.__name__
            spec = ""
            if hasattr(layer, "n_keep"):
                spec += f"n_keep={layer.n_keep} "
            if hasattr(layer, "n_drop"):
                spec += f"n_drop: {layer.n_drop}"
            lines.append(f"  [{i:02d}] {name} {spec}")
        lines.append(")")
        return "\n".join(lines)

    def _bijector_fn(self, params):
        means, log_scales = unstack(params, -1) # happens on non-surjective layer in heirarchical
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _base_distribution_fn(self, n_dim):
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
            reinterpreted_batch_ndims=1,
        )
        return base_distribution

    def _conditioner_fn(self, input_dim, output_dim, **kwargs):
        if self.nflow_config.conditioner == "mlp":
            return make_mlp(
                [self.nflow_config.conditioner_n_neurons]
              * self.nflow_config.conditioner_n_layers
              + [output_dim],
            )
        elif self.nflow_config.conditioner == "transformer":
            return make_transformer(
                {
                    'output_dim': output_dim,
                    'num_heads': 4,
                    'num_layers': 4,
                    'key_size': 32,
                    'dropout_rate': 0.1,
                    'widening_factor': 4,
                    **kwargs
                }
            )
        elif self.nflow_config.conditioner == 'made':
            return MADE(
                input_size=input_dim,
                hidden_layer_sizes=[self.nflow_config.conditioner_n_neurons]
                    * self.nflow_config.conditioner_n_layers,
                n_params=2,
                w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                b_init=jnp.zeros,
                activation=jax.nn.tanh
            )

        logging.fatal("didnt find correct conditioner type")
        raise ValueError("didnt find correct conditioner type")

    def _decoder_fn(
            self, 
            n_dimension: int, 
            decoder_distribution: str,
            eps: float = 1e-8,
            integer_transform: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    ):
        decoder_params_lookup = {
            'gaussian': 2,
            'nb': 2,
            'poisson': 1,
            'students_t': 3
        }
        self.n_decoder_params = decoder_params_lookup[decoder_distribution]

        decoder_net = make_mlp(
            [self.nflow_config.decoder_n_neurons] * self.nflow_config.decoder_n_layers
          + [n_dimension * self.n_decoder_params],
            activation=jax.nn.tanh,
        )

        def _fn(z):
            # we get a distribution per pixel, e.g. 500 latent pixels => 500 mus for Gaussian
            params = decoder_net(z)
            if decoder_distribution == 'gaussian':
                mu, log_scale = jnp.split(params, 2, -1)
                return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)), 1)

            elif decoder_distribution == 'nb':
                mu_raw, r_raw = jnp.split(params, 2, axis=-1)
                mu = jax.nn.softplus(mu_raw) + eps
                r  = jax.nn.softplus(r_raw) + eps
                dist = NegBinomDist(mu, r)
                return IndependentWrapper(dist, reinterpreted_batch_ndims=1)

            elif decoder_distribution == 'poisson':
                log_lambda = params
                log_lambda = jnp.clip(log_lambda, -10, 10)
                lam = jnp.exp(log_lambda)
                dist = PoissonDist(lam)
                return IndependentWrapper(
                    dist, 
                    reinterpreted_batch_ndims=1, 
                    integer_transform=integer_transform
                )

            elif decoder_distribution == 'students_t':
                mu, log_scale, log_df = jnp.split(params, 3, -1)
                scale = jnp.exp(log_scale)
                df = jnp.exp(log_df) + 2.0     # keep df > 2 for finite variance
                dist = StudentT(df=df, loc=mu, scale=scale) # type: ignore
                return IndependentWrapper(dist, reinterpreted_batch_ndims=1)

            else:
                raise Exception('Unrecognised decoder distribution.')

        return _fn

    def _flow(self, method: str, **kwargs):
        return self._make_flow(method, **kwargs)

    def _get_surjective_layer(
            self, 
            name: str
    ) -> (
            AffineMaskedAutoregressiveInferenceFunnel
          | RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel
        ):
        name_to_class = {
            'affine_MAF': AffineMaskedAutoregressiveInferenceFunnel,
            'rational_quadratic_MAF': (
                RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel
            )
        }
        return name_to_class[name]
    
    def _healpix_funnel(
            self, 
            surjective_layer_type: (
                AffineMaskedAutoregressiveInferenceFunnel
              | RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel
            ), 
            layers: list, 
            blocks: list[tuple[int, int]],
            in_dim: int,
            integer_transform: Optional[
                Callable[
                    [int | Literal['all']],
                    Callable[[jnp.ndarray], jnp.ndarray]
                ]
            ],
            one_and_done: bool = False,
            maf_extension: Optional[int] = None
    ) -> tuple[list, int]:
        dim = in_dim

        if one_and_done:
            assert maf_extension is None
            n_kp = blocks[-1][0]
            n_drp = dim - n_kp
            assert n_kp + n_drp == dim
            blocks = [(n_kp, n_drp)]

        if maf_extension:
            assert maf_extension % 2 == 0
        else:
            maf_extension = 0

        for lvl, (n_keep, n_drop) in enumerate(blocks):
            # n_keep and n_dropped split data into y_minus and y_plus, as in
            # y_plus, y_minus = y[..., : self.n_keep], y[..., self.n_keep :]
            surjective_layer = surjective_layer_type( # type: ignore
                n_keep=n_keep,
                decoder=self._decoder_fn(
                    n_dimension=n_drop,
                    decoder_distribution=self.nflow_config.decoder_distribution,
                    integer_transform=(
                        integer_transform(lvl) if integer_transform else None
                    )
                ),
                conditioner=self._conditioner_fn(
                    input_dim=n_keep,
                    output_dim=2 * n_keep
                )
            )
            setattr(surjective_layer, 'n_drop', n_drop) # for __repr__
            layers.append(surjective_layer)
            dim = n_keep

            # bijective extension
            for _ in range(maf_extension):
                layers.append(
                    MaskedAutoregressive(
                        bijector_fn=self._bijector_fn,
                        conditioner=self._conditioner_fn(
                            input_dim=dim,
                            output_dim=2 * dim
                        )
                    )
                )
                layers = self._reverse_perm(layers, dim)
                
        out_dim = dim
        return layers, out_dim

    def _reverse_perm(self, layers: list, dim: int) -> list:
        order = jnp.arange(dim)
        order = order[::-1]
        layers.append(Permutation(order, 1))
        return layers

    def _make_flow(self, method: str, **kwargs):
        assert self.nflow_config.architecture is not None

        cur_dim = self.target_ndim
        self.layers = []

        for layer in self.nflow_config.architecture:
            if layer == 'MAF':
                self.layers.append(
                    MaskedAutoregressive(
                        bijector_fn=self._bijector_fn,
                        conditioner=self._conditioner_fn(
                            input_dim=cur_dim,
                            output_dim=2 * cur_dim
                        )
                    )
                )
                self.layers = self._reverse_perm(self.layers, cur_dim)

            elif layer == 'healpix_funnel':
                assert self.nflow_config.surjective_layer_type is not None
                assert self.nflow_config.funnel_one_and_done is not None
                assert self.nflow_config.funnel_maf_extension is not None
                assert self.blocks is not None

                surjective_layer_type = self._get_surjective_layer(
                    self.nflow_config.surjective_layer_type
                )
                # add surjective healpix funnel
                self.layers, cur_dim = self._healpix_funnel(
                    surjective_layer_type, 
                    self.layers, 
                    self.blocks, 
                    cur_dim, 
                    integer_transform=None,
                    one_and_done=self.nflow_config.funnel_one_and_done,
                    maf_extension=self.nflow_config.funnel_maf_extension
                )
                # don't add an extra perm --- added by the maf extension
                if self.nflow_config.funnel_maf_extension == 0:
                    self.layers = self._reverse_perm(self.layers, cur_dim)

            elif layer == 'surjective_MAF':
                assert self.nflow_config.data_reduction_factor is not None
                n_keep = int(self.nflow_config.data_reduction_factor * cur_dim)
                n_drop = cur_dim - n_keep
                assert n_keep + n_drop == cur_dim

                surjective_layer_type = self._get_surjective_layer(
                    self.nflow_config.surjective_layer_type
                )
                surjective_layer = surjective_layer_type( # type: ignore
                    n_keep=n_keep,
                    decoder=self._decoder_fn(
                        n_dimension=n_drop,
                        decoder_distribution=self.nflow_config.decoder_distribution,
                        integer_transform=None
                    ),
                    conditioner=self._conditioner_fn(
                        input_dim=n_keep,
                        output_dim=2 * n_keep
                    )
                )
                setattr(surjective_layer, 'n_drop', n_drop) # for __repr__
                self.layers.append(surjective_layer)
                cur_dim = n_keep
                self.layers = self._reverse_perm(self.layers, cur_dim)

            else:
                raise Exception(f'Layer {layer} not recognised.')

        # remove redundant perm at end of stack
        self.layers = self.layers[:-1]

        chain = Chain(self.layers)
        td = TransformedDistribution(self._base_distribution_fn(cur_dim), chain)
        return td(method, **kwargs)

