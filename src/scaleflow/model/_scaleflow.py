import functools
import os
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import anndata as ad
import cloudpickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from ott.neural.methods.flows import dynamics

from scaleflow import _constants
from scaleflow._types import ArrayLike, Layers_separate_input_t, Layers_t
from scaleflow.data import (
    AnnDataLocation,
    DataManager,
    GroupedAnnbatchSampler,
    GroupedDistribution,
    PredictionSampler,
    SamplerABC,
    SourceCache,
    ValidationSampler,
)
from scaleflow.model._utils import _write_predictions
from scaleflow.networks import _velocity_field
from scaleflow.plotting import _utils
from scaleflow.solvers import _eqm, _genot, _otfm
from scaleflow.training._callbacks import BaseCallback
from scaleflow.training._trainer import CellFlowTrainer
from scaleflow.utils import match_linear

from scaleflow.data import SamplerABC

__all__ = ["ScaleFlow"]


class ScaleFlow:
    """Perturbation-prediction model (Flow Matching / Optimal Transport) over on-disk data.

    Cells are streamed from an :class:`annbatch.DatasetCollection` via
    :class:`~scaleflow.data.GroupedAnnbatchSampler` (annbatch ``ClassSampler``), so training
    scales beyond memory and across multiple datasets. Use :meth:`prepare_data` (a collection
    path), :meth:`prepare_model`, then :meth:`train`.

    Parameters
    ----------
        solver
            Solver to use for training. Either ``'otfm'``, ``'genot'`` or ``'eqm'``.
    """

    def __init__(self, solver: Literal["otfm", "genot", "eqm"] = "otfm"):
        if solver == "otfm":
            self._solver_class = _otfm.OTFlowMatching
            self._vf_class = _velocity_field.ConditionalVelocityField
        elif solver == "genot":
            self._solver_class = _genot.GENOT
            self._vf_class = _velocity_field.GENOTConditionalVelocityField
        elif solver == "eqm":
            self._solver_class = _eqm.EquilibriumMatching
            self._vf_class = _velocity_field.EquilibriumVelocityField
        else:
            raise ValueError(f"Unknown solver: {solver}. Must be 'otfm', 'genot', or 'eqm'.")
        self._dataloader: SamplerABC | None = None
        self._trainer: CellFlowTrainer | None = None
        self._validation_data: dict[str, GroupedDistribution] = {"predict_kwargs": {}}
        self._solver: _otfm.OTFlowMatching | _genot.GENOT | _eqm.EquilibriumMatching | None = None
        self._condition_dim: int | None = None
        self._vf: (
            _velocity_field.ConditionalVelocityField
            | _velocity_field.GENOTConditionalVelocityField
            | _velocity_field.EquilibriumVelocityField
            | None
        ) = None

    def prepare_data(
        self,
        collection: Any,
        *,
        dist_flag_key: str,
        src_dist_keys: Sequence[str],
        tgt_dist_keys: Sequence[str] | dict[str, Sequence[str]],
        rep_keys: dict[str, str] | None = None,
        rep_path: str | None = None,
        rep_dict: Mapping[str, Any] | None = None,
        data_location: AnnDataLocation | None = None,
        extra_rep_keys: dict[str, tuple[str, str]] | None = None,
        verbose: bool = False,
    ) -> None:
        """Prepare grouped-distribution training metadata from an on-disk store.

        Cells are streamed on demand by annbatch from ``collection`` (an
        :class:`annbatch.DatasetCollection` or a path to one); no in-memory AnnData is
        required and ``X`` is never fully materialized. Source (control) and target
        (perturbed) cells are grouped into distributions from the collection's ``obs``,
        and condition embeddings are read from a separate ``uns`` store (``rep_path``)
        or an in-memory mapping (``rep_dict``) — a :class:`annbatch.DatasetCollection`
        does not round-trip ``uns``.

        Parameters
        ----------
        collection
            An :class:`annbatch.DatasetCollection` or a path to one.
        dist_flag_key
            Boolean ``obs`` column that is ``True`` for control (source) cells.
        src_dist_keys
            ``obs`` columns that define the source (control) populations.
        tgt_dist_keys
            ``obs`` columns that define the target (perturbation) conditions.
        rep_keys
            Mapping ``{covariate_column: rep_store_key}`` selecting the embedding for
            each covariate. Columns without an entry are used numerically (e.g. dosage).
        rep_path
            Path to a zarr/h5 ``uns``-shaped store whose top-level keys are the
            ``rep_keys`` values (each a ``{label: embedding}`` mapping).
        rep_dict
            Already-loaded representations, as an alternative to ``rep_path``.
        data_location
            Which array to stream as the cell representation. Defaults to ``X``.
        extra_rep_keys
            Optional extra embeddings keyed by a new condition name; see
            :class:`~scaleflow.data.DataManager`.
        verbose
            Whether to print timing information.

        Returns
        -------
        Updates :attr:`data_manager` and :attr:`train_data`.
        """
        if data_location is None:
            data_location = AnnDataLocation().X
        self._dm = DataManager(
            dist_flag_key=dist_flag_key,
            src_dist_keys=list(src_dist_keys),
            # grouped combinations ({group: [cols]}) pass through; a flat sequence stays a list
            tgt_dist_keys=(
                {g: list(cols) for g, cols in tgt_dist_keys.items()}
                if isinstance(tgt_dist_keys, dict)
                else list(tgt_dist_keys)
            ),
            rep_keys=dict(rep_keys) if rep_keys else {},
            data_location=data_location,
            extra_rep_keys=extra_rep_keys,
        )
        if rep_dict is None and rep_path is None:
            rep_dict = {}
        self._train_collection = collection
        self.train_data = self._dm.prepare_data_from_collection(
            collection, rep_path=rep_path, rep_dict=rep_dict, verbose=verbose
        )

    def make_dataloader(
        self,
        *,
        batch_size: int = 1024,
        chunk_size: int,
        preload_nchunks: int | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> GroupedAnnbatchSampler:
        """Build an annbatch-backed training sampler over the prepared collection.

        ``chunk_size`` is the annbatch ``ClassSampler`` read-slice size; it must be <= every
        trained condition's cell count and ``chunk_size * preload_nchunks`` must be divisible
        by ``batch_size`` (see :class:`~scaleflow.data.GroupedAnnbatchSampler`). The collection
        must be written sorted by condition (see
        :func:`~scaleflow.data.write_sorted_collection`).
        """
        if self.train_data is None or getattr(self, "_train_collection", None) is None:
            raise ValueError("Call `prepare_data` first.")
        return GroupedAnnbatchSampler(
            self._train_collection,
            self.train_data,
            batch_size=batch_size,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            seed=seed,
            **kwargs,
        )

    @staticmethod
    def _feature_dim(collection: Any) -> int:
        """Read the cell-feature dimension from a collection without loading X."""
        import zarr

        from scaleflow.data._annbatch_sampler import _open_collection

        coll = _open_collection(collection)
        g = next(iter(coll))
        x = g["X"]
        if isinstance(x, zarr.Array):
            return int(x.shape[-1])
        return int(ad.io.sparse_dataset(x).shape[-1])

    def prepare_validation_data(
        self,
        name: str,
        val_data: GroupedDistribution,
        collection: Any | None = None,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Register a validation split for use during training.

        Parameters
        ----------
        name
            Key under which to store this validation set in :attr:`validation_data`.
        val_data
            A :class:`~scaleflow.data.GroupedDistribution` (e.g. a val/test split) whose
            cells are streamed from ``collection``.
        collection
            The :class:`annbatch.DatasetCollection` (or path) backing ``val_data``.
            Defaults to the training collection passed to :meth:`prepare_data`.
        n_conditions_on_log_iteration
            Number of conditions to validate on at each logged iteration (all if None).
        n_conditions_on_train_end
            Number of conditions to validate on at the end of training (all if None).
        predict_kwargs
            Keyword arguments forwarded to the solver's ``predict`` during validation.
        """
        if self.train_data is None:
            raise ValueError("Call `prepare_data` before `prepare_validation_data`.")
        if not isinstance(val_data, GroupedDistribution):
            raise ValueError("`val_data` must be a `GroupedDistribution` (e.g. a split).")
        self._validation_data[name] = {
            "gd": val_data,
            "collection": collection if collection is not None else getattr(self, "_train_collection", None),
            "n_log": n_conditions_on_log_iteration,
            "n_end": n_conditions_on_train_end,
        }
        self._validation_data.setdefault("predict_kwargs", {})
        if predict_kwargs:
            self._validation_data["predict_kwargs"].update(predict_kwargs)

    def prepare_model(
        self,
        sample_batch: dict[str, Any] | None = None,
        max_combination_length: int = 1,
        condition_mode: Literal["deterministic", "stochastic"] = "deterministic",
        regularization: float = 0.0,
        pooling: Literal["mean", "attention_token", "attention_seed"] = "attention_token",
        pooling_kwargs: dict[str, Any] = types.MappingProxyType({}),
        layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(default_factory=lambda: []),
        layers_after_pool: Layers_t = dc_field(default_factory=lambda: []),
        condition_embedding_dim: int = 256,
        cond_output_dropout: float = 0.9,
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
        use_phenotype_predictor: bool = False,
        phenotype_hidden_dims: Sequence[int] = (256, 128, 64),
        phenotype_dropout: float = 0.0,
        phenotype_output_dim: int = 1,
        loss_weight_gex: float = 1.0,
        loss_weight_functional: float = 1.0,
        seed=0,
    ) -> None:
        """Prepare the model for training.

        This function sets up the neural network architecture and specificities of the
        :attr:`solver`. When :attr:`solver` is an instance of :class:`scaleflow.solvers._genot.GENOT`,
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

            - :class:`scaleflow.networks.TokenAttentionPooling` if ``'pooling'`` is
              ``'attention_token'``.
            - :class:`scaleflow.networks.SeedAttentionPooling` if ``'pooling'`` is ``'attention_seed'``.

        layers_before_pool
            Layers applied to the condition embeddings before pooling. Can be of type

            - :class:`tuple` with elements corresponding to dictionaries with keys:

                - ``'layer_type'`` of type :class:`str` indicating the type of the layer, can be
                  ``'mlp'`` or ``'self_attention'``.
                - Further keyword arguments for the layer type :class:`scaleflow.networks.MLPBlock` or
                  :class:`scaleflow.networks.SelfAttentionBlock`.

            - :class:`dict` with keys corresponding to perturbation covariate keys, and values
              correspondinng to the above mentioned tuples.

        layers_after_pool
            Layers applied to the condition embeddings after pooling, and before applying the last
            layer of size ``'condition_embedding_dim'``. Should be of type :class:`tuple` with
            elements corresponding to dictionaries with keys:

            - ``'layer_type'`` of type :class:`str` indicating the type of the layer, can be
              ``'mlp'`` or ``'self_attention'``.
            - Further keys depend on the layer type, either for :class:`scaleflow.networks.MLPBlock` or
              for :class:`scaleflow.networks.SelfAttentionBlock`.

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
            :func:`scaleflow.networks.utils.sinusoidal_time_encoder`.
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
                  of the :class:`scaleflow.networks.MLPBlock` processing the source cell.
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
            :func:`scaleflow.utils.match_linear`.
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

        - :attr:`scaleflow.model.ScaleFlow.velocity_field` - an instance of the
          :class:`scaleflow.networks.ConditionalVelocityField`.
        - :attr:`scaleflow.model.ScaleFlow.solver` - an instance of :class:`scaleflow.solvers.OTFlowMatching`
          or :class:`scaleflow.solvers.GENOT`.
        - :attr:`scaleflow.model.ScaleFlow.trainer` - an instance of the
          :class:`scaleflow.training.CellFlowTrainer`.
        """

        # Store the seed for use in train method
        self._seed = seed

        if sample_batch is not None:
            sample_conditions = sample_batch["condition"]
            self._data_dim = sample_batch["src_cell_data"].shape[-1]
        else:
            # derive sizing from the prepared data without streaming a batch
            sample_conditions = next(iter(self.train_data.data.conditions.values()))
            self._data_dim = self._feature_dim(self._train_collection)

        # Combination handling is data-driven: each covariate's set length is read from the
        # condition shapes (axis 1). A covariate with a set length > 1 is a perturbation
        # *combination* that the encoder pools; covariates with set length 1 cannot be
        # concatenated into a >1 pool, so they are encoded as not-pooled context covariates
        # (concatenated after pooling). max_combination_length is the data's true set width
        # -- no padding is used; it only sizes the init dummy condition.
        set_sizes = {k: int(np.asarray(v).shape[1]) for k, v in sample_conditions.items()}
        max_combination_length = max(set_sizes.values()) if set_sizes else 1
        if max_combination_length > 1:
            covariates_not_pooled = [k for k, s in set_sizes.items() if s == 1]
        else:
            covariates_not_pooled = []

        if condition_mode == "stochastic":
            if regularization == 0.0:
                raise ValueError("Stochastic condition embeddings require `regularization`>0.")

        condition_encoder_kwargs = condition_encoder_kwargs or {}
        if (
            self._solver_class == _otfm.OTFlowMatching or self._solver_class == _eqm.EquilibriumMatching
        ) and vf_kwargs is not None:
            raise ValueError("For `solver='otfm'` or `solver='eqm'`, `vf_kwargs` must be `None`.")
        if self._solver_class == _genot.GENOT:
            if vf_kwargs is None:
                vf_kwargs = {"genot_source_dims": [1024, 1024, 1024], "genot_source_dropout": 0.0}
            else:
                assert isinstance(vf_kwargs, dict)
                assert "genot_source_dims" in vf_kwargs
                assert "genot_source_dropout" in vf_kwargs
        else:
            vf_kwargs = {}
        solver_kwargs = solver_kwargs or {}
        probability_path = probability_path or {"constant_noise": 0.0}

        if self._solver_class == _eqm.EquilibriumMatching:
            self.vf = self._vf_class(
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
        else:
            self.vf = self._vf_class(
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
                condition_encoder_kwargs=condition_encoder_kwargs,
                act_fn=vf_act_fn,
                time_freqs=time_freqs,
                time_max_period=time_max_period,
                time_encoder_dims=time_encoder_dims,
                time_encoder_dropout=time_encoder_dropout,
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
                **vf_kwargs,
            )

        probability_path, noise = next(iter(probability_path.items()))
        if probability_path == "constant_noise":
            probability_path = dynamics.ConstantNoiseFlow(noise)
        elif probability_path == "bridge":
            probability_path = dynamics.BrownianBridge(noise)
        else:
            raise NotImplementedError(
                f"The key of `probability_path` must be `'constant_noise'` or `'bridge'` but found {probability_path}."
            )

        phenotype_predictor = None
        if use_phenotype_predictor:
            from scaleflow.networks import PhenotypePredictor

            phenotype_predictor = PhenotypePredictor(
                hidden_dims=phenotype_hidden_dims,
                dropout_rate=phenotype_dropout,
                output_dim=phenotype_output_dim,
            )

        # Get sample conditions from first target distribution
        # Conditions are stored as nested dicts: {col_name: array}

        if self._solver_class == _otfm.OTFlowMatching:
            self._solver = self._solver_class(
                vf=self.vf,
                match_fn=match_fn,
                probability_path=probability_path,
                phenotype_predictor=phenotype_predictor,
                loss_weight_gex=loss_weight_gex,
                loss_weight_functional=loss_weight_functional,
                optimizer=optimizer,
                conditions=sample_conditions,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        elif self._solver_class == _eqm.EquilibriumMatching:
            # EqM doesn't use probability_path, only match_fn
            self._solver = self._solver_class(
                vf=self.vf,
                match_fn=match_fn,
                phenotype_predictor=phenotype_predictor,
                loss_weight_gex=loss_weight_gex,
                loss_weight_functional=loss_weight_functional,
                optimizer=optimizer,
                conditions=sample_conditions,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        elif self._solver_class == _genot.GENOT:
            self._solver = self._solver_class(
                vf=self.vf,
                data_match_fn=match_fn,
                probability_path=probability_path,
                source_dim=self._data_dim,
                target_dim=self._data_dim,
                optimizer=optimizer,
                conditions=sample_conditions,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Solver must be an instance of OTFlowMatching, EquilibriumMatching, or GENOT, got {type(self.solver)}"
            )

        self._trainer = CellFlowTrainer(solver=self.solver, predict_kwargs=self.validation_data["predict_kwargs"])  # type: ignore[arg-type]

    def train(
        self,
        num_iterations: int,
        batch_size: int = 1024,
        chunk_size: int | None = None,
        preload_nchunks: int | None = None,
        valid_freq: int = 1000,
        callbacks: Sequence[BaseCallback] = [],
        monitor_metrics: Sequence[str] = [],
        train_dataloader: SamplerABC | None = None,
        val_dataloader: SamplerABC | None = None,
    ) -> None:
        """Train the model.

        Note
        ----
        A low value of ``'valid_freq'`` results in long training
        because predictions are time-consuming compared to training steps.

        For multi-task training with functional assays, create a custom dataloader
        that returns batches with the appropriate 'task' field ('gex' or 'functional').

        Parameters
        ----------
        num_iterations
            Number of iterations to train the model.
        batch_size
            Batch size.
        chunk_size
            annbatch ``ClassSampler`` read-slice size for the default training sampler.
            Required when ``train_dataloader`` is not supplied. Must be <= every trained
            condition's cell count and ``chunk_size * preload_nchunks`` must be divisible by
            ``batch_size``. The collection must be written sorted by condition (see
            :func:`~scaleflow.data.write_sorted_collection`).
        preload_nchunks
            Number of chunks loaded per ``ClassSampler`` window (``None`` auto-picks a valid
            value).
        valid_freq
            Frequency of validation.
        callbacks
            Callbacks to perform at each validation step. There are two types of callbacks:
            - Callbacks for computations should inherit from
              :class:`~scaleflow.training.ComputationCallback` see e.g. :class:`scaleflow.training.Metrics`.
            - Callbacks for logging should inherit from :class:`~scaleflow.training.LoggingCallback` see
              e.g. :class:`~scaleflow.training.WandbLogger`.
        monitor_metrics
            Metrics to monitor.
        train_dataloader
            Optional pre-built training sampler. If ``None``, a
            :class:`~scaleflow.data.GroupedAnnbatchSampler` is built over the prepared
            collection (``chunk_size`` is then required).
        val_dataloader
            Optional mapping of validation samplers. If ``None``, one
            :class:`~scaleflow.data.ValidationSampler` is built per registered validation split.

        Returns
        -------
        Updates the following fields:

        - :attr:`scaleflow.model.ScaleFlow.dataloader` - the training dataloader.
        - :attr:`scaleflow.model.ScaleFlow.solver` - the trained solver.
        """

        if self.trainer is None:
            raise ValueError("Model not initialized. Please call `prepare_model` first.")

        # Build the annbatch-backed training sampler over the prepared collection if
        # the caller did not provide one.
        if train_dataloader is None:
            if chunk_size is None:
                raise ValueError(
                    "`chunk_size` is required when training builds the default annbatch sampler. "
                    "It is the ClassSampler read-slice size and must be <= every trained condition's "
                    "cell count (see GroupedAnnbatchSampler). Pass `chunk_size=...` to `train`, or supply "
                    "a `train_dataloader`."
                )
            train_dataloader = self.make_dataloader(
                batch_size=batch_size,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                seed=getattr(self, "_seed", 0),
            )
        self._dataloader = train_dataloader

        # Build annbatch-backed validation samplers from registered validation splits.
        if val_dataloader is None:
            val_dataloader = {
                name: ValidationSampler(
                    info["collection"],
                    info["gd"],
                    n_conditions_on_log_iteration=info["n_log"],
                    n_conditions_on_train_end=info["n_end"],
                )
                for name, info in self._validation_data.items()
                if name != "predict_kwargs"
            }
        self._trainer.predict_kwargs = self._validation_data.get("predict_kwargs", {})

        self._solver = self.trainer.train(
            dataloader=train_dataloader,
            num_iterations=num_iterations,
            valid_freq=valid_freq,
            valid_loaders=val_dataloader,
            callbacks=callbacks,
            monitor_metrics=monitor_metrics,
        )

    def predict(
        self,
        data: GroupedDistribution,
        collection: Any | None = None,
        rng: ArrayLike | None = None,
        **kwargs: Any,
    ) -> dict[tuple[str, ...], ArrayLike]:
        """Predict perturbation responses for the conditions of a GroupedDistribution.

        Source (control) cells are streamed from ``collection`` and pushed forward under
        each target condition's embedding. Prediction is per-condition (conditions may
        have different numbers of source cells).

        Parameters
        ----------
        data
            A :class:`~scaleflow.data.GroupedDistribution` defining the conditions to
            predict (e.g. a held-out test split).
        collection
            The :class:`annbatch.DatasetCollection` (or path) backing ``data``. Defaults
            to the training collection passed to :meth:`prepare_data`.
        rng
            Random number generator; only used in ``'stochastic'`` condition mode.
        kwargs
            Keyword arguments forwarded to the solver's ``predict``.

        Returns
        -------
        A :class:`dict` mapping each condition key (label tuple) to its predicted sample
        representation.
        """
        if self.solver is None or not self.solver.is_trained:
            raise ValueError("Model not trained. Please call `train` first.")

        coll = collection if collection is not None else getattr(self, "_train_collection", None)
        if coll is None:
            raise ValueError("No collection available; pass `collection` or call `prepare_data` first.")

        pred_loader = PredictionSampler(coll, data)
        batch = pred_loader.sample()
        src = batch["source"]
        condition = batch["condition"]
        # per-condition prediction (jax.tree.map flattens up to `src`, so each condition
        # dict is treated as a single leaf passed as `condition`).
        return jax.tree.map(
            functools.partial(self.solver.predict, rng=rng, **kwargs),
            src,
            condition,
        )

    def predict_covariates(
        self,
        covariate_data: pd.DataFrame,
        collection: Any | None = None,
        rep_dict: dict[str, Any] | None = None,
        rep_path: str | None = None,
        rng: ArrayLike | None = None,
        **kwargs: Any,
    ) -> dict[tuple[str, ...], ArrayLike]:
        """Predict responses for arbitrary (possibly unseen) covariate combinations.

        Builds a condition embedding for each unique row of ``covariate_data`` and pushes
        forward the matching trained control population (matched by ``src_dist_keys``
        label, i.e. "set the perturbations to control to find the source").

        Parameters
        ----------
        covariate_data
            DataFrame whose columns include ``src_dist_keys`` and ``tgt_dist_keys``.
        collection
            Collection backing the control (source) populations; defaults to the training
            collection.
        rep_dict / rep_path
            Representations (embeddings) for the covariates.
        rng
            Random number generator; only used in ``'stochastic'`` condition mode.
        kwargs
            Forwarded to the solver's ``predict``.

        Returns
        -------
        Mapping ``{condition_key: prediction}`` where ``condition_key`` is the
        ``(src_dist_keys + tgt_dist_keys)`` label tuple.
        """
        if self.solver is None or not self.solver.is_trained:
            raise ValueError("Model not trained. Please call `train` first.")
        if self.train_data is None:
            raise ValueError("Call `prepare_data` first (control populations come from the training data).")
        coll = collection if collection is not None else getattr(self, "_train_collection", None)
        if coll is None:
            raise ValueError("No collection available; pass `collection` or call `prepare_data` first.")

        conditions = self._dm.get_condition_data(covariate_data, rep_dict=rep_dict, rep_path=rep_path)
        source_cache = SourceCache(coll, self.train_data.data.src_dist_to_rows)
        label_to_src = {
            tuple(str(x) for x in lbl): int(idx)
            for idx, lbl in self.train_data.annotation.src_dist_idx_to_labels.items()
        }
        n_src = len(self._dm.src_dist_keys)

        predict_fn = functools.partial(self.solver.predict, rng=rng, **kwargs)
        out: dict[tuple[str, ...], ArrayLike] = {}
        for cond_key, cond_dict in conditions.items():
            src_label = tuple(cond_key[:n_src])
            if src_label not in label_to_src:
                raise ValueError(f"No trained control population for source label {src_label}.")
            src_cells = source_cache.cells(label_to_src[src_label])
            out[cond_key] = predict_fn(src_cells, cond_dict)
        return out

    def get_condition_embedding(
        self,
        covariate_data: pd.DataFrame,
        rep_dict: dict[str, Any] | None = None,
        rep_path: str | None = None,
        condition_id_key: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get the learnt embedding (mean and log-variance) of conditions.

        Builds condition embeddings for arbitrary covariate combinations via
        :meth:`~scaleflow.data.DataManager.get_condition_data` and encodes each with the
        trained :class:`~scaleflow.networks.ConditionEncoder`.

        Parameters
        ----------
        covariate_data
            A :class:`~pandas.DataFrame` whose columns include the ``src_dist_keys`` and
            ``tgt_dist_keys`` registered in :attr:`data_manager`.
        rep_dict / rep_path
            Representations (embeddings); a separate ``uns`` store path or an in-memory
            mapping. If both are :obj:`None`, an empty mapping is used.
        condition_id_key
            Optional column in ``covariate_data`` to use as the condition key.

        Returns
        -------
        A :class:`tuple` of :class:`~pandas.DataFrame` with the mean and log-variance of
        the condition embeddings, indexed by condition key.
        """
        if self.solver is None or not self.solver.is_trained:
            raise ValueError("Model not trained. Please call `train` first.")

        conditions = self._dm.get_condition_data(
            covariate_data,
            rep_dict=rep_dict,
            rep_path=rep_path,
            condition_id_key=condition_id_key,
        )

        means: dict[Any, ArrayLike] = {}
        logvars: dict[Any, ArrayLike] = {}
        for cond_key, cond in conditions.items():
            mean, logvar = self.solver.get_condition_embedding(cond)
            means[cond_key] = np.asarray(mean)[0]
            logvars[cond_key] = np.asarray(logvar)[0]

        df_mean = pd.DataFrame.from_dict(means, orient="index")
        df_var = pd.DataFrame.from_dict(logvars, orient="index")
        return df_mean, df_var

        return df_mean, df_var

    def save(
        self,
        dir_path: str,
        file_prefix: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save the model.

        Pickles the :class:`~scaleflow.model.ScaleFlow` object.

        Parameters
        ----------
            dir_path
                Path to a directory, defaults to current directory
            file_prefix
                Prefix to prepend to the file name.
            overwrite
                Overwrite existing data or not.

        Returns
        -------
            :obj:`None`
        """
        file_name = (
            f"{file_prefix}_{self.__class__.__name__}.pkl"
            if file_prefix is not None
            else f"{self.__class__.__name__}.pkl"
        )
        file_dir = os.path.join(dir_path, file_name) if dir_path is not None else file_name

        if not overwrite and os.path.exists(file_dir):
            raise RuntimeError(f"Unable to save to an existing file `{file_dir}` use `overwrite=True` to overwrite it.")
        with open(file_dir, "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(
        cls,
        filename: str,
    ) -> "ScaleFlow":
        """
        Load a :class:`~scaleflow.model.ScaleFlow` model from a saved instance.

        Parameters
        ----------
            filename
                Path to the saved file.

        Returns
        -------
        Loaded instance of the model.
        """
        # Check if filename is a directory
        file_name = os.path.join(filename, f"{cls.__name__}.pkl") if os.path.isdir(filename) else filename

        with open(file_name, "rb") as f:
            model = cloudpickle.load(f)

        if type(model) is not cls:
            raise TypeError(f"Expected the model to be type of `{cls}`, found `{type(model)}`.")
        return model

    @property
    def adata(self) -> ad.AnnData:
        """The :class:`~anndata.AnnData` object used for training."""
        return self._adata

    @property
    def solver(self) -> _otfm.OTFlowMatching | _genot.GENOT | _eqm.EquilibriumMatching | None:
        """The solver."""
        return self._solver

    @property
    def dataloader(self) -> SamplerABC | None:
        """The dataloader used for training."""
        return self._dataloader

    @property
    def trainer(self) -> CellFlowTrainer | None:
        """The trainer used for training."""
        return self._trainer

    @property
    def validation_data(self) -> dict[str, GroupedDistribution]:
        """The validation data."""
        return self._validation_data

    @property
    def data_manager(self) -> DataManager:
        """The data manager, initialised with :attr:`scaleflow.model.ScaleFlow.adata`."""
        return self._dm

    @property
    def velocity_field(
        self,
    ) -> (
        _velocity_field.ConditionalVelocityField
        | _velocity_field.GENOTConditionalVelocityField
        | _velocity_field.EquilibriumVelocityField
        | None
    ):
        """The conditional velocity field."""
        return self._vf

    @property
    def train_data(self) -> GroupedDistribution | None:
        """The training data."""
        return self._train_data

    @train_data.setter
    def train_data(self, data: GroupedDistribution) -> None:
        """Set the training data."""
        if not isinstance(data, GroupedDistribution):
            raise ValueError(f"Expected `data` to be an instance of `GroupedDistribution`, found `{type(data)}`.")
        self._train_data = data

    @velocity_field.setter  # type: ignore[attr-defined,no-redef]
    def velocity_field(self, vf: _velocity_field.ConditionalVelocityField) -> None:
        """Set the velocity field."""
        if not isinstance(vf, _velocity_field.ConditionalVelocityField):
            raise ValueError(f"Expected `vf` to be an instance of `ConditionalVelocityField`, found `{type(vf)}`.")
        self._vf = vf

    @property
    def condition_mode(self) -> Literal["deterministic", "stochastic"]:
        """The mode of the encoder."""
        return self.velocity_field.condition_mode
