import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
from cellflow.networks._utils import FilmBlock, MLPBlock, ResNetBlock, sinusoidal_time_encoder
from flax import linen as nn
from flax.training import train_state

from cellflow._types import Layers_separate_input_t, Layers_t
from cellflow.networks._set_encoders import ConditionEncoder
from cellflow.networks._velocity_field import (
    ConditionalVelocityField as _CFConditionalVelocityField,
)
from cellflow.networks._velocity_field import (
    GENOTConditionalVelocityField as _CFGENOTConditionalVelocityField,
)

__all__ = ["ConditionalVelocityField", "GENOTConditionalVelocityField", "EquilibriumVelocityField"]


class ConditionalVelocityField(_CFConditionalVelocityField):
    """Velocity field extending :class:`cellflow.networks.ConditionalVelocityField`.

    Inherits all fields, the condition encoder, CFG nulling (``condition_null`` +
    ``_maybe_null_*``), ``get_condition_embedding``, ``create_train_state``, and the
    ``setup``/``__call__`` scaffolding. Adds two scaleflow-only features by overriding
    the conditioning hooks: an ``'adaln_zero'`` conditioning mode (per-cell AdaLN-Zero
    modulation replacing the decoder) and an optional cell self-attention transformer
    applied ``'before_condition'`` or ``'after_condition'``.

    Legacy docstring (inherited parameters):

    Parameters
    ----------
        output_dim
            Dimensionality of the output.
        max_combination_length
            Maximum number of covariates in a combination.
        condition_mode
            Mode of the encoder, should be one of:

            - ``'deterministic'``: Learns condition encoding point-wise.
            - ``'stochastic'``: Learns a Gaussian distribution for representing conditions.

        regularization
            Regularization strength in the latent space:

            - For deterministic mode, it is the strength of the L2 regularization.
            - For stochastic mode, it is the strength of the KL divergence regularization.

        condition_embedding_dim
            Dimensions of the condition embedding.
        covariates_not_pooled
            Covariates that will escape pooling (should be identical across all set elements).
        pooling
            Pooling method.
        pooling_kwargs
            Keyword arguments for the pooling method.
        layers_before_pool
            Layers before pooling. Either a sequence of tuples with layer type and parameters or
            a dictionary with input-specific layers.
        layers_after_pool
            Layers after pooling.
        cond_output_dropout
            Dropout rate for the last layer of the condition encoder.
        condition_encoder_kwargs
            Keyword arguments for the condition encoder.
        act_fn
            Activation function.
        time_freqs
            Frequency of the cyclical time encoding.
        time_max_period
            Controls the minimum frequency of the time embeddings.
        time_encoder_dims
            Dimensions of the time embedding.
        time_encoder_dropout
            Dropout rate for the time embedding.
        hidden_dims
            Dimensions of the hidden layers.
        hidden_dropout
            Dropout rate for the hidden layers.
        conditioning
            Conditioning method, should be one of:

            - ``'concatenation'``: Concatenate the time, data, and condition embeddings.
            - ``'film'``: Use FiLM conditioning, i.e. learn FiLM weights from time and condition embedding
              to scale the data embeddings.
            - ``'resnet'``: Use residual conditioning.

        conditioning_kwargs
            Keyword arguments for the conditioning method.
        decoder_dims
            Dimensions of the output layers (or attention dimensions for adaln_zero).
            For adaln_zero, MLP dimensions are automatically set to match hidden_dims[-1].
        decoder_dropout
            Dropout rate for the output layers.
        layer_norm_before_concatenation
            If :obj:`True`, applies layer normalization before concatenating
            the embedded time, embedded data, and condition embeddings.
        linear_projection_before_concatenation
            If :obj:`True`, applies a linear projection before concatenating
            the embedded time, embedded data.

    Returns
    -------
        Output of the neural vector field.
    """

    conditioning: Literal["concatenation", "film", "resnet", "adaln_zero"] = "concatenation"
    cell_transformer_layers: int = 0
    cell_transformer_heads: int = 8
    cell_transformer_dim: int = 128
    cell_transformer_dropout: float = 0.1
    cell_transformer_mode: Literal["before_condition", "after_condition"] = "before_condition"

    def _adaln_cond_dim(self) -> int:
        """Size of the vector that modulates the adaln blocks: ``(t, condition)`` here."""
        return self.time_encoder_dims[-1] + self.condition_embedding_dim

    def _setup_conditioning(self, conditioning_kwargs: dict[str, Any]) -> None:
        """Add the ``'adaln_zero'`` mode and an optional cell transformer; delegate the rest."""
        if self.conditioning == "adaln_zero":
            from scaleflow.networks._utils import build_adaln_blocks

            # Per-cell AdaLN-Zero: modulate each cell by its OWN conditioning vector. No cross-cell
            # attention — the velocity field must be a per-cell function, and cells within a
            # batch have different flow-times t, so they must not attend to each other.
            self.adaln_blocks = build_adaln_blocks(
                decoder_dims=self.decoder_dims,
                cond_dim=self._adaln_cond_dim(),
                decoder_dropout=self.decoder_dropout,
                act_fn=self.act_fn,
                conditioning_kwargs=conditioning_kwargs,
            )
        else:
            super()._setup_conditioning(conditioning_kwargs)

        if self.cell_transformer_layers > 0:
            from cellflow.networks._utils import SelfAttentionBlock

            self.cell_transformer = SelfAttentionBlock(
                num_heads=[self.cell_transformer_heads] * self.cell_transformer_layers,
                qkv_dim=[self.cell_transformer_dim] * self.cell_transformer_layers,
                dropout_rate=self.cell_transformer_dropout,
                transformer_block=True,
                layer_norm=True,
                act_fn=self.act_fn,
            )

    def _encode_x(self, x_t: jnp.ndarray, squeeze: bool, train: bool) -> jnp.ndarray:
        """Encode ``x_t``, optionally passing it through the cell transformer first."""
        x_encoded = self.x_encoder(x_t, training=train)
        if self.cell_transformer_layers > 0 and self.cell_transformer_mode == "before_condition":
            if squeeze:
                x_encoded_expanded = jnp.expand_dims(x_encoded, 0)
                x_encoded = self.cell_transformer(x_encoded_expanded, mask=None, training=train)
                x_encoded = jnp.squeeze(x_encoded, 0)
            else:
                x_encoded_expanded = jnp.expand_dims(x_encoded, 0) if x_encoded.ndim == 1 else x_encoded
                x_encoded = self.cell_transformer(x_encoded_expanded, mask=None, training=train)
        return x_encoded

    def _combine_and_decode(
        self,
        t_encoded: jnp.ndarray,
        x_encoded: jnp.ndarray,
        cond_embedding: jnp.ndarray,
        squeeze: bool,
        train: bool,
        x_0_encoded: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Combine/decode, adding ``'adaln_zero'`` and the ``'after_condition'`` cell transformer."""
        concat_inputs, conditioning_vec = self._conditioning_signals(t_encoded, x_encoded, cond_embedding, x_0_encoded)
        if self.conditioning == "concatenation":
            out = jnp.concatenate(concat_inputs, axis=-1)
        elif self.conditioning == "film":
            out = self.film_block(x_encoded, conditioning_vec)
        elif self.conditioning == "resnet":
            out = self.resnet_block(x_encoded, conditioning_vec)
        elif self.conditioning == "adaln_zero":
            out = x_encoded
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}.")

        if self.cell_transformer_layers > 0 and self.cell_transformer_mode == "after_condition":
            if squeeze:
                out_expanded = jnp.expand_dims(out, 0)
                out = self.cell_transformer(out_expanded, mask=None, training=train)
                out = jnp.squeeze(out, 0)
            else:
                out_expanded = jnp.expand_dims(out, 0) if out.ndim == 1 else out
                out = self.cell_transformer(out_expanded, mask=None, training=train)

        if self.conditioning == "adaln_zero":
            from scaleflow.networks._utils import apply_adaln

            # Modulate each cell by its OWN conditioning vector — see build_adaln_blocks.
            return apply_adaln(
                adaln_blocks=self.adaln_blocks,
                output_layer=self.output_layer,
                out=out,
                conditioning_vec=conditioning_vec,
                squeeze=squeeze,
                train=train,
            )

        out = self.decoder(out, training=train)
        return self.output_layer(out)


class GENOTConditionalVelocityField(_CFGENOTConditionalVelocityField):
    """GENOT velocity field, extending :class:`cellflow.networks.GENOTConditionalVelocityField`.

    Adds the ``'adaln_zero'`` conditioning mode (per-cell AdaLN-Zero modulation by the
    ``(t, x_0, condition)`` vector — the same signals GENOT's film/resnet already use,
    with the source folded in), reusing the shared adaln helpers. Also carries the
    ``cell_transformer_*`` fields so the uniform model constructor can build it (GENOT
    does not use a cell transformer, so they are accepted and ignored).
    """

    conditioning: Literal["concatenation", "film", "resnet", "adaln_zero"] = "concatenation"
    cell_transformer_layers: int = 0
    cell_transformer_heads: int = 8
    cell_transformer_dim: int = 128
    cell_transformer_dropout: float = 0.1
    cell_transformer_mode: Literal["before_condition", "after_condition"] = "before_condition"

    def _setup_conditioning(self, conditioning_kwargs: dict[str, Any]) -> None:
        """Add the ``'adaln_zero'`` mode (modulated by ``(t, x_0, condition)``); delegate the rest."""
        if self.conditioning == "adaln_zero":
            from scaleflow.networks._utils import build_adaln_blocks

            self.adaln_blocks = build_adaln_blocks(
                decoder_dims=self.decoder_dims,
                cond_dim=self.time_encoder_dims[-1] + self.genot_source_dims[-1] + self.condition_embedding_dim,
                decoder_dropout=self.decoder_dropout,
                act_fn=self.act_fn,
                conditioning_kwargs=conditioning_kwargs,
            )
        else:
            super()._setup_conditioning(conditioning_kwargs)

    def _combine_and_decode(
        self,
        t_encoded: jnp.ndarray,
        x_encoded: jnp.ndarray,
        cond_embedding: jnp.ndarray,
        squeeze: bool,
        train: bool,
        x_0_encoded: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Route ``'adaln_zero'`` through the shared adaln helper; delegate the rest."""
        if self.conditioning == "adaln_zero":
            from scaleflow.networks._utils import apply_adaln

            _, conditioning_vec = self._conditioning_signals(t_encoded, x_encoded, cond_embedding, x_0_encoded)
            return apply_adaln(
                adaln_blocks=self.adaln_blocks,
                output_layer=self.output_layer,
                out=x_encoded,
                conditioning_vec=conditioning_vec,
                squeeze=squeeze,
                train=train,
            )
        return super()._combine_and_decode(t_encoded, x_encoded, cond_embedding, squeeze, train, x_0_encoded)


class EquilibriumVelocityField(ConditionalVelocityField):
    """Gradient field for Equilibrium Matching — a time-less :class:`ConditionalVelocityField`.

    Reuses everything (setup, condition encoder, cell transformer, the
    concat/film/resnet/adaln_zero conditioning, and the helpers) but drops the time
    encoder: EqM's field is a function of ``(x, condition)`` only (the gamma interpolation
    lives in the solver, not the field), so it modulates/concatenates by the condition
    alone. Its ``__call__`` therefore takes no ``t`` and does no classifier-free guidance.
    """

    def _setup_time(self) -> None:
        """EqM has no time encoder."""

    def _adaln_cond_dim(self) -> int:
        """EqM modulates the adaln blocks by the condition only (no time)."""
        return self.condition_embedding_dim

    def _conditioning_signals(
        self,
        t_encoded: jnp.ndarray,
        x_encoded: jnp.ndarray,
        cond_embedding: jnp.ndarray,
        x_0_encoded: jnp.ndarray | None = None,
    ) -> tuple[tuple[jnp.ndarray, ...], jnp.ndarray]:
        """No time (or source): concatenate/modulate by the condition alone."""
        return (x_encoded, cond_embedding), cond_embedding

    def __call__(
        self,
        x: jnp.ndarray,
        cond: dict[str, jnp.ndarray],
        encoder_noise: jnp.ndarray,
        train: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        squeeze = x.ndim == 1
        cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
        if self.condition_mode == "deterministic":
            cond_embedding = cond_mean
        else:
            cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)
        cond_embedding = self.layer_cond_output_dropout(cond_embedding, deterministic=not train)

        x_encoded = self._encode_x(x, squeeze, train)
        x_encoded = self.layer_norm_x(x_encoded)
        cond_embedding = self.layer_norm_condition(cond_embedding)

        if squeeze:
            cond_embedding = jnp.squeeze(cond_embedding)
        elif cond_embedding.shape[0] != x.shape[0]:
            cond_embedding = jnp.tile(cond_embedding, (x.shape[0], 1))

        out = self._combine_and_decode(None, x_encoded, cond_embedding, squeeze, train)
        return out, cond_mean, cond_logvar

    def create_train_state(
        self,
        rng: jax.Array,
        optimizer: optax.OptState,
        input_dim: int,
        conditions: dict[str, jnp.ndarray],
    ) -> train_state.TrainState:
        """Create the training state (no time variable)."""
        x = jnp.ones((1, input_dim))
        encoder_noise = jnp.ones((1, self.condition_embedding_dim))
        cond = {
            pert_cov: jnp.ones((1, self.max_combination_length, condition.shape[-1]))
            for pert_cov, condition in conditions.items()
        }
        params_rng, condition_encoder_rng = jax.random.split(rng, 2)
        params = self.init(
            {"params": params_rng, "condition_encoder": condition_encoder_rng},
            x=x,
            cond=cond,
            encoder_noise=encoder_noise,
            train=False,
        )["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
