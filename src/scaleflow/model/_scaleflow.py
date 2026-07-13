import types
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from cellflow.model._cellflow import CellFlow as _CFCellFlow

from cellflow._types import ArrayLike, Layers_separate_input_t, Layers_t
from scaleflow._compat import BrownianBridge, ConstantNoiseFlow
from scaleflow.networks import _velocity_field
from scaleflow.solvers import SOLVER_REGISTRY, EquilibriumMatching, GENOT, OTFlowMatching
from scaleflow.training._trainer import CellFlowTrainer
from cellflow.utils import match_linear

__all__ = ["ScaleFlow"]


class ScaleFlow(_CFCellFlow):
    """CellFlow with scaleflow's velocity fields and ``sf_``-prefixed solvers.

    Subclasses :class:`cellflow.model.CellFlow` to reuse all of its data preparation
    (the in-memory ``prepare_data`` and the ``prepare_annbatch_data`` / ``dagloader``
    streaming path), ``prepare_validation_data``, ``train``, ``predict``, save/load, and
    the properties. scaleflow overrides only the model-building bits: the ``sf_*`` solver
    selection (``__init__``) and the velocity-field construction (``prepare_model``:
    adaln_zero / cell transformer). The scaleflow-specific classifier-free-guidance
    validation sweep is preserved because ``prepare_model`` still builds scaleflow's
    :class:`scaleflow.training.CellFlowTrainer` (the inherited ``train`` uses it).

    NOTE — functionality dropped by inheriting cellflow's ``train``: scaleflow's old
    ``train`` accepted ``train_dataloader``/``val_dataloader``/``num_workers``/
    ``prefetch_factor``/``log_every`` (pass a custom sampler + threading knobs).
    cellflow's ``train`` instead builds the loader from the prepared data
    (``TrainSampler``/``OOCTrainSampler``, or the ``DAGLoader`` for the annbatch path).

    Parameters
    ----------
        solver
            Solver to use for training. Either ``'sf_otfm'``, ``'sf_genot'`` or ``'sf_eqm'``.
    """

    def __init__(self, solver: Literal["sf_otfm", "sf_genot", "sf_eqm"] = "sf_otfm"):
        # scaleflow's solvers live under ``sf_*`` keys in cellflow's shared registry (so they
        # don't clobber cellflow's own ``otfm``/``genot``); only those keys are selectable here.
        if not solver.startswith("sf_") or solver not in SOLVER_REGISTRY:
            available = sorted(k for k in SOLVER_REGISTRY if k.startswith("sf_"))
            raise ValueError(f"Unknown solver {solver!r}. Available: {available}.")
        super().__init__(adata=None, solver=solver)

    def prepare_model(
        self,
        sample_batch: dict[str, Any],
        max_combination_length: int = 1,
        condition_mode: Literal["deterministic", "stochastic"] = "deterministic",
        regularization: float = 0.0,
        pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token",
        pooling_kwargs: dict[str, Any] = types.MappingProxyType({}),
        layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: []),
        layers_after_pool: Layers_t = dc_field(default_factory=lambda: []),
        condition_embedding_dim: int = 256,
        cond_output_dropout: float = 0.9,
        condition_dropout_prob: float = 0.0,   # classifier-free guidance: prob of nulling the condition
        condition_null: Literal["zero_embedding", "mask_value"] = "zero_embedding",  # how CFG builds the null
        condition_encoder_kwargs: dict[str, Any] | None = None,
        pool_sample_covariates: bool = True,
        time_freqs: int = 1024,
        time_max_period: int | None = 10000,
        time_encoder_dims: Sequence[int] = (2048, 2048, 2048),
        time_encoder_dropout: float = 0.0,
        hidden_dims: Sequence[int] = (2048, 2048, 2048),
        hidden_dropout: float = 0.0,
        cell_transformer_layers: int = 0,
        cell_transformer_heads: int = 8,
        cell_transformer_dim: int = 128,
        cell_transformer_dropout: float = 0.1,
        cell_transformer_mode: Literal["before_condition", "after_condition"] = "before_condition",
        conditioning: Literal["concatenation", "film", "resnet"] = "concatenation",
        conditioning_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {}),
        decoder_dims: Sequence[int] = (4096, 4096, 4096),
        decoder_dropout: float = 0.0,
        vf_act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu,
        vf_kwargs: dict[str, Any] | None = None,
        probability_path: dict[Literal["constant_noise", "bridge"], float] | None = None,
        match_fn: Callable[[ArrayLike, ArrayLike], ArrayLike] = match_linear,
        optimizer: optax.GradientTransformation = optax.MultiSteps(optax.adam(5e-5), 20),
        solver_kwargs: dict[str, Any] | None = None,
        layer_norm_before_concatenation: bool = False,
        linear_projection_before_concatenation: bool = False,
        seed=0,
    ) -> None:
        """Prepare the model for training.

        This function sets up the neural network architecture and specificities of the
        :attr:`solver`. When :attr:`solver` is an instance of :class:`scaleflow.solvers.GENOT`,
        the following arguments have to be passed to ``'condition_encoder_kwargs'``:


        Parameters
        ----------
        condition_mode
            Mode of the encoder, should be one of:

            - ``'deterministic'``: Learns condition encoding point-wise.
            - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.

        regularization
            Regularization strength in the latent space:

            - For deterministic mode, it is the strength of the L2 regularization.
            - For stochastic mode, it is the strength of the VAE regularization.

        pooling
            Pooling method, should be one of:

            - ``'mean'``: Aggregates combinations of covariates by the mean of their
              learned embeddings.
            - ``'attention_token'``: Aggregates combinations of covariates by an attention
              mechanism with a class token.
            - ``'attention_seed'``: Aggregates combinations of covariates by seed attention.

        pooling_kwargs
            Keyword arguments for the pooling method corresponding to:

            - :class:`cellflow.networks.TokenAttentionPooling` if ``'pooling'`` is
              ``'attention_token'``.
            - :class:`cellflow.networks.SeedAttentionPooling` if ``'pooling'`` is ``'attention_seed'``.

        layers_before_pool
            Layers applied to the condition embeddings before pooling. Can be of type

            - :class:`tuple` with elements corresponding to dictionaries with keys:

                - ``'layer_type'`` of type :class:`str` indicating the type of the layer, can be
                  ``'mlp'`` or ``'self_attention'``.
                - Further keyword arguments for the layer type :class:`cellflow.networks.MLPBlock` or
                  :class:`cellflow.networks.SelfAttentionBlock`.

            - :class:`dict` with keys corresponding to perturbation covariate keys, and values
              correspondinng to the above mentioned tuples.

        layers_after_pool
            Layers applied to the condition embeddings after pooling, and before applying the last
            layer of size ``'condition_embedding_dim'``. Should be of type :class:`tuple` with
            elements corresponding to dictionaries with keys:

            - ``'layer_type'`` of type :class:`str` indicating the type of the layer, can be
              ``'mlp'`` or ``'self_attention'``.
            - Further keys depend on the layer type, either for :class:`cellflow.networks.MLPBlock` or
              for :class:`cellflow.networks.SelfAttentionBlock`.

        condition_embedding_dim
            Dimensions of the condition embedding, i.e. the last layer of the
            :class:`scaleflow.networks.ConditionEncoder`.
        cond_output_dropout
            Dropout rate for the last layer of the :class:`scaleflow.networks.ConditionEncoder`.
        condition_encoder_kwargs
            Keyword arguments for the :class:`scaleflow.networks.ConditionEncoder`.
        pool_sample_covariates
            Whether to include sample covariates in the pooling.
        time_freqs
            Frequency of the sinusoidal time encoding
            (:func:`ott.neural.networks.layers.sinusoidal_time_encoder`).
        time_max_period
            Controls the frequency of the time embeddings, see
            :func:`cellflow.networks._utils.sinusoidal_time_encoder`.
        time_encoder_dims
            Dimensions of the layers processing the time embedding in
            :attr:`scaleflow.networks.ConditionalVelocityField.time_encoder`.
        time_encoder_dropout
            Dropout rate for the :attr:`scaleflow.networks.ConditionalVelocityField.time_encoder`.
        hidden_dims
            Dimensions of the layers processing the input to the velocity field
            via :attr:`scaleflow.networks.ConditionalVelocityField.x_encoder`.
        hidden_dropout
            Dropout rate for :attr:`scaleflow.networks.ConditionalVelocityField.x_encoder`.
        conditioning
            Conditioning method, should be one of:

            - ``'concatenation'``: Concatenate the time, data, and condition embeddings.
            - ``'film'``: Use FiLM conditioning, i.e. learn FiLM weights from time and condition embedding
              to scale the data embeddings.
            - ``'resnet'``: Use residual conditioning.

        conditioning_kwargs
            Keyword arguments for the conditioning method.
        decoder_dims
            Dimensions of the output layers in
            :attr:`scaleflow.networks.ConditionalVelocityField.decoder`.
        decoder_dropout
            Dropout rate for the output layer
            :attr:`scaleflow.networks.ConditionalVelocityField.decoder`.
        vf_act_fn
            Activation function of the :class:`scaleflow.networks.ConditionalVelocityField`.
        vf_kwargs
            Additional keyword arguments for the solver-specific vector field.
            For instance, when ``'solver==genot'``, the following keyword argument can be passed:

                - ``'genot_source_dims'`` of type :class:`tuple` with the dimensions
                  of the :class:`cellflow.networks.MLPBlock` processing the source cell.
                - ``'genot_source_dropout'`` of type :class:`float` indicating the dropout rate
                  for the source cell processing.
        probability_path
            Probability path to use for training. Should be a :class:`dict` of the form

            - ``'{"constant_noise": noise_val'``
            - ``'{"bridge": noise_val}'``

            If :obj:`None`, defaults to ``'{"constant_noise": 0.0}'``.
        match_fn
            Matching function between unperturbed and perturbed cells. Should take as input source
            and target data and return the optimal transport matrix, see e.g.
            :func:`cellflow.utils.match_linear`.
        optimizer
            Optimizer used for training.
        solver_kwargs
            Keyword arguments for the solver :class:`scaleflow.solvers.OTFlowMatching` or
            :class:`scaleflow.solvers.GENOT`.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data, and embedded condition.
        seed
            Random seed.

        Returns
        -------
        Updates the following fields:

        - :attr:`scaleflow.model.CellFlow.velocity_field` - an instance of the
          :class:`scaleflow.networks.ConditionalVelocityField`.
        - :attr:`scaleflow.model.CellFlow.solver` - an instance of :class:`scaleflow.solvers.OTFlowMatching`
          or :class:`scaleflow.solvers.GENOT`.
        - :attr:`scaleflow.model.CellFlow.trainer` - an instance of the
          :class:`scaleflow.training.CellFlowTrainer`.
        """
        # Store the seed for use in train method
        self._seed = seed

        sample_conditions = sample_batch["condition"]
        self._data_dim = sample_batch["src_cell_data"].shape[-1]

        if condition_mode == "stochastic":
            if regularization == 0.0:
                raise ValueError("Stochastic condition embeddings require `regularization`>0.")

        condition_encoder_kwargs = condition_encoder_kwargs or {}
        # Each velocity field owns which solver-specific `vf_kwargs` it accepts (GENOT needs
        # source-processing dims; the others take none), mirroring cellflow's normalization hook.
        vf_kwargs = self._vf_class._normalize_vf_kwargs(vf_kwargs)
        covariates_not_pooled = [] if pool_sample_covariates else self._dm.sample_covariates
        solver_kwargs = solver_kwargs or {}
        probability_path = probability_path or {"constant_noise": 0.0}

        vf_args = dict(
            output_dim=self._data_dim,
            max_combination_length=max_combination_length,
            condition_mode=condition_mode,
            regularization=regularization,
            condition_embedding_dim=condition_embedding_dim,
            covariates_not_pooled=covariates_not_pooled,
            pooling=pooling,
            pooling_kwargs=pooling_kwargs,
            layers_before_pool=layers_before_pool,
            layers_after_pool=layers_after_pool,
            cond_output_dropout=cond_output_dropout,
            condition_dropout_prob=condition_dropout_prob,
            condition_encoder_kwargs=condition_encoder_kwargs,
            act_fn=vf_act_fn,
            hidden_dims=hidden_dims,
            hidden_dropout=hidden_dropout,
            cell_transformer_layers=cell_transformer_layers,
            cell_transformer_heads=cell_transformer_heads,
            cell_transformer_dim=cell_transformer_dim,
            cell_transformer_dropout=cell_transformer_dropout,
            cell_transformer_mode=cell_transformer_mode,
            conditioning=conditioning,
            conditioning_kwargs=conditioning_kwargs,
            decoder_dims=decoder_dims,
            decoder_dropout=decoder_dropout,
            layer_norm_before_concatenation=layer_norm_before_concatenation,
            linear_projection_before_concatenation=linear_projection_before_concatenation,
        )
        # EqM's velocity field is reduced: it has no time encoder and no classifier-free-guidance
        # nulling, so those args go only to the time-conditioned (OTFM/GENOT) velocity fields.
        if self._solver_class is not EquilibriumMatching:
            vf_args.update(
                time_freqs=time_freqs,
                time_max_period=time_max_period,
                time_encoder_dims=time_encoder_dims,
                time_encoder_dropout=time_encoder_dropout,
                condition_null=condition_null,
            )
        self.vf = self._vf_class(**vf_args, **vf_kwargs)

        probability_path, noise = next(iter(probability_path.items()))
        if probability_path == "constant_noise":
            probability_path = ConstantNoiseFlow(noise)
        elif probability_path == "bridge":
            probability_path = BrownianBridge(noise)
        else:
            raise NotImplementedError(
                f"The key of `probability_path` must be `'constant_noise'` or `'bridge'` but found {probability_path}."
            )

        # Get sample conditions from first target distribution
        # Conditions are stored as nested dicts: {col_name: array}

        # Each solver owns how it names its match function / needs data dims, mirroring cellflow.
        solver_args = dict(
            vf=self.vf,
            optimizer=optimizer,
            conditions=sample_conditions,
            rng=jax.random.PRNGKey(seed),
            **self._solver_class._match_kwargs(match_fn=match_fn, data_dim=self._data_dim),
            **solver_kwargs,
        )
        # EqM interpolates via gamma and has no probability path.
        if self._solver_class is not EquilibriumMatching:
            solver_args["probability_path"] = probability_path
        self._solver = self._solver_class(**solver_args)

        self._trainer = CellFlowTrainer(solver=self.solver, predict_kwargs=self.validation_data["predict_kwargs"])  # type: ignore[arg-type]

