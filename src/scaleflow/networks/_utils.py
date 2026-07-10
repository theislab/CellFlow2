from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from cellflow._types import Layers_t
from cellflow.networks._utils import (
    BaseModule,
    FilmBlock,
    MLPBlock,
    ResNetBlock,
    sinusoidal_time_encoder,
)
from flax import linen as nn
from flax.linen import initializers

__all__ = [
    "SelfAttention",
    "SeedAttentionPooling",
    "TokenAttentionPooling",
    "MLPBlock",
    "FilmBlock",
    "ResNetBlock",
    "SelfAttentionBlock",
    "AdaLNZeroBlock",
    "AdaLNModulation",
    "sinusoidal_time_encoder",
]












class SelfAttention(BaseModule):
    """Self-attention layer

    Self-attention layer that can optionally be followed by a FC layer with residual connection,
    making it a transformer block.

    Parameters
    ----------
    num_heads
        Number of heads.
    qkv_dim
        Dimensionality of the query, key, and value.
    dropout_rate
        Dropout rate.
    transformer_block
        Whether to make it a transformer block (adds FC layer with residual connection).
    layer_norm
        Whether to use layer normalization
    """

    num_heads: int = 8
    qkv_dim: int = 64
    ff_dim: int | None = None
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ):
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch_size, set_size, input_dim)`` or
            ``(batch_size, input_dim)``.
        mask
            Mask tensor of shape ``(batch_size, 1 | num_heads, set_size, set_size)``.
        training
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape ``(batch_size, set_size, input_dim)``.
        """
        squeeze = x.ndim == 2
        x = jnp.expand_dims(x, 1) if squeeze else x

        # self-attention
        z = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_dim,
            dropout_rate=self.dropout_rate,
        )(x, mask=mask, deterministic=not training)

        if self.transformer_block:
            # attention residual connection
            z = nn.Dropout(self.dropout_rate)(z, deterministic=not training)
            z = z + x
            if self.layer_norm:
                z = nn.LayerNorm()(z)
            # position-wise feed-forward: expand to ff_dim (default 4x input_dim, the canonical
            # transformer expansion — NOT qkv_dim, which is the attention dim), then back to input_dim.
            d_ff = self.ff_dim if self.ff_dim is not None else 4 * z.shape[-1]
            z_ = self.act_fn(nn.Dense(d_ff)(z))
            z_ = nn.Dense(z.shape[-1])(z_)
            z_ = nn.Dropout(self.dropout_rate)(z_, deterministic=not training)
            z = z + z_
            if self.layer_norm:
                z = nn.LayerNorm()(z)

        return z.squeeze(1) if squeeze else z


class SelfAttentionBlock(BaseModule):
    """
    Several self-attention (+ optional FC layer) layers stacked together.

    Parameters
    ----------
    num_heads
        Number of heads for each layer.
    qkv_dim
        Dimensionality of the query, key, and value for each layer.
    dropout_rate
        Dropout rate.
    transformer_block
        Whether to make layers transformer blocks (adds FC layer with residual connection).
    layer_norm
        Whether to use layer normalization.

    Returns
    -------
    Output tensor of shape (batch_size, set_size, input_dim).
    """

    num_heads: Sequence[int] | int
    qkv_dim: Sequence[int] | int
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    ff_dim: int | None = None
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    def __post_init__(self) -> None:
        """Initialize the module."""
        super().__post_init__()
        if not isinstance(self.num_heads, Sequence):
            self.num_heads = [self.num_heads]
        if not isinstance(self.qkv_dim, Sequence):
            self.qkv_dim = [self.qkv_dim]
        if len(self.num_heads) != len(self.qkv_dim):
            raise ValueError("The number of specified layers should be the same for num_heads and qkv_dims.")

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch_size, set_size, input_dim).
        mask : Optional[jnp.ndarray]
            Mask tensor of shape (batch_size, 1 | num_heads, set_size, set_size).
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape (batch_size, set_size, input_dim).
        """
        z = x
        for num_heads, qkv_dim in zip(self.num_heads, self.qkv_dim, strict=False):  # type: ignore[arg-type]
            z = SelfAttention(
                num_heads=num_heads,
                qkv_dim=qkv_dim,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                transformer_block=self.transformer_block,
                layer_norm=self.layer_norm,
                act_fn=self.act_fn,
            )(z, mask, training)
        return z


class SeedAttentionPooling(BaseModule):
    """
    Pooling by multi-head attention with a trainable seed.

    Parameters
    ----------
    num_heads
        Number of heads.
    v_dim
        Dimensionality of the value.
    seed_dim
        Dimensionality of the seed.
    dropout_rate
        Dropout rate.
    transformer_block
        Whether to make it a transformer block (adds FC layer with residual connection).
    layer_norm
        Whether to use layer normalization.
    act_fn
        Activation function.

    References
    ----------
    :cite:`vaswani:17`
    """

    num_heads: int = 8
    v_dim: int = 64
    seed_dim: int = 64
    dropout_rate: float = 0.0
    transformer_block: bool = False
    layer_norm: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ):
        """
        Apply the pooling by multi-head attention.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch_size, set_size, input_dim)``.
        mask
            Mask tensor of shape ``(batch_size, 1, set_size, set_size)``.
        training
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape ``(batch_size, input_dim)``.
        """
        # trainable seed
        S = self.param("S", initializers.xavier_uniform(), (1, 1, self.seed_dim))
        S = jnp.tile(S, (x.shape[0], 1, 1))

        # multi-head attention
        Q = nn.Dense(self.v_dim)(S)
        K, V = nn.Dense(self.v_dim)(x), nn.Dense(self.v_dim)(x)
        Q_ = jnp.concatenate(jnp.split(Q, self.num_heads, axis=2), axis=0)
        K_ = jnp.concatenate(jnp.split(K, self.num_heads, axis=2), axis=0)
        V_ = jnp.concatenate(jnp.split(V, self.num_heads, axis=2), axis=0)
        A = jnp.matmul(Q_, K_.transpose(0, 2, 1)) / jnp.sqrt(self.v_dim)
        A = jnp.matmul(Q_, K_.transpose(0, 2, 1)) / jnp.sqrt(self.v_dim)
        if mask is not None:
            # mask from (batch_, 1 | num_heads, set_, set_) to (batch_ * num_heads, 1, set_)
            mask = jnp.repeat(mask[:, 0, [0], :], self.num_heads, axis=0)
            A = jnp.where(mask, A, -1e9)
        A = nn.softmax(A)
        A = jnp.matmul(A, V_)

        if self.transformer_block:
            # query residual connection
            O = jnp.concatenate(jnp.split(Q_ + A, self.num_heads, axis=0), axis=2)
            O = nn.Dropout(rate=self.dropout_rate)(O, deterministic=not training)
            if self.layer_norm:
                O = nn.LayerNorm()(O)
            # FC layer with residual connection
            O_ = self.act_fn(nn.Dense(self.v_dim)(O))
            O_ = nn.Dropout(rate=self.dropout_rate)(O_, deterministic=not training)
            O = O + O_
            if self.layer_norm:
                O = nn.LayerNorm()(O)
        else:
            O = jnp.concatenate(jnp.split(A, self.num_heads, axis=0), axis=2)

        return O.squeeze(1)


class TokenAttentionPooling(BaseModule):
    """
    Multi-head attention which aggregates sets by learning a token.

    A learnable ``[CLS]`` token is prepended to the set and ``num_layers`` self-attention layers are
    applied over ``{CLS, *tokens}`` (BERT/ViT-style); the CLS token's final state is returned as the
    aggregate. ``num_layers=1`` with ``transformer_block=False`` reproduces the original single-attention
    pooling. Set ``num_layers>=2`` (+ ``transformer_block=True``) to make the condition encoder a
    transformer where every token attends to every other before the CLS read-out.

    Parameters
    ----------
    num_heads
        Number of attention heads (per layer).
    qkv_dim
        Dimensionality of the query, key, and value (per layer).
    dropout_rate
        Dropout rate.
    num_layers
        Number of stacked self-attention layers over ``{CLS, *tokens}``.
    transformer_block
        Whether each layer is a full transformer block (adds a feed-forward sublayer with residual).
    layer_norm
        Whether to use layer normalization inside each transformer block.
    act_fn
        Activation function.
    """

    num_heads: int = 8
    qkv_dim: int = 64
    dropout_rate: float = 0.0
    num_layers: int = 1
    transformer_block: bool = False
    layer_norm: bool = False
    ff_dim: int | None = None
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape (batch_size, set_size, input_dim).
        mask
            Mask tensor of shape (batch_size, 1 | num_heads, set_size, set_size).
        training
            Whether the model is in training mode.

        Returns
        -------
        Output tensor of shape ``(batch_size, input_dim)``.
        """
        # add token
        token_shape = (len(x), 1)
        class_token = nn.Embed(num_embeddings=1, features=x.shape[-1])(jnp.int32(jnp.zeros(token_shape)))
        z = jnp.concatenate((class_token, x), axis=-2)
        token_mask = jnp.ones((x.shape[0], 1, x.shape[1] + 1, x.shape[1] + 1))
        token_mask = token_mask.at[:, :, 1:, 1:].set(mask)
        cls_token_to_data = mask[0, 0, :, :].sum(axis=0) > 0
        token_mask = token_mask.at[:, :, 0, 1:].set(cls_token_to_data)
        token_mask = token_mask.at[:, :, 1:, 0].set(cls_token_to_data)

        # transformer over {CLS, *tokens}: num_layers self-attention layers, then read out the CLS.
        if self.num_layers == 1 and not self.transformer_block:
            # backward-compatible single-attention path: identical module structure (hence Flax param
            # tree) to the original TokenAttentionPooling, so historical (pre-refactor) checkpoints
            # restore unchanged. num_layers>=2 or transformer_block=True use the SelfAttentionBlock.
            z = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.qkv_dim,
                dropout_rate=self.dropout_rate,
            )(z, mask=token_mask, deterministic=not training)
        else:
            z = SelfAttentionBlock(
                num_heads=[self.num_heads] * self.num_layers,
                qkv_dim=[self.qkv_dim] * self.num_layers,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                transformer_block=self.transformer_block,
                layer_norm=self.layer_norm,
                act_fn=self.act_fn,
            )(z, token_mask, training)

        # only continue with the CLS token (position 0)
        return z[:, 0, :]


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization modulation from DiT (Diffusion Transformers).

    Produces scale, shift, and gate parameters for modulating layer normalization
    based on conditioning information (time + condition embeddings).

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden features to modulate.
    cond_dim : int
        Dimension of the conditioning vector.
    use_gate : bool
        Whether to output gate parameters (for adaLN-Zero).
    act_fn : Callable
        Activation function.

    References
    ----------
    Peebles & Xie (2023). Scalable Diffusion Models with Transformers.
    https://arxiv.org/abs/2212.09748
    """

    hidden_dim: int
    cond_dim: int
    use_gate: bool = True
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, conditioning: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute modulation parameters from conditioning.

        Parameters
        ----------
        conditioning : jnp.ndarray
            Conditioning vector (e.g., concatenated time and condition embeddings).
            Shape: (batch, cond_dim)

        Returns
        -------
        tuple
            (scale, shift, gate) each of shape (batch, hidden_dim).
            If use_gate=False, gate is None.
        """
        num_outputs = 3 if self.use_gate else 2
        modulation = nn.Dense(
            num_outputs * self.hidden_dim, kernel_init=initializers.zeros, bias_init=initializers.zeros
        )(self.act_fn(conditioning))

        if self.use_gate:
            scale, shift, gate = jnp.split(modulation, 3, axis=-1)
            return scale, shift, gate
        else:
            scale, shift = jnp.split(modulation, 2, axis=-1)
            return scale, shift, None


class AdaLNZeroBlock(BaseModule):
    """
    Adaptive Layer Normalization with zero initialization (adaLN-Zero).

    Applies layer normalization with learned scale and shift modulated by
    conditioning, followed by the main operation (MLP or Attention), and
    a learned gate. All modulation parameters are initialized to zero.

    This is the key building block from the DiT paper for conditioning
    diffusion/flow models.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden features.
    num_heads : int | None
        Number of attention heads (if using self-attention).
    qkv_dim : int | None
        Dimension of Q, K, V in attention.
    mlp_dim : int | None
        Explicit MLP hidden dimension. If None, uses mlp_ratio * hidden_dim.
    mlp_ratio : float
        Ratio of MLP hidden dim to input dim (default 4.0). Only used if mlp_dim is None.
    dropout_rate : float
        Dropout rate.
    use_attention : bool
        If True, uses self-attention. If False, uses MLP.
    act_fn : Callable
        Activation function.

    References
    ----------
    Peebles & Xie (2023). Scalable Diffusion Models with Transformers.
    https://arxiv.org/abs/2212.09748
    """

    hidden_dim: int
    cond_dim: int
    num_heads: int = 8
    qkv_dim: int | None = None
    mlp_dim: int | None = None
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    use_attention: bool = True
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        conditioning: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Apply adaLN-Zero block.

        Parameters
        ----------
        x : jnp.ndarray
            Input features of shape (batch, seq_len, hidden_dim) or (batch, hidden_dim).
        conditioning : jnp.ndarray
            Conditioning vector of shape (batch, cond_dim).
        mask : jnp.ndarray | None
            Attention mask.
        training : bool
            Whether in training mode.

        Returns
        -------
        jnp.ndarray
            Output features of same shape as input.
        """
        qkv_features = self.qkv_dim or self.hidden_dim
        mlp_hidden_dim = self.mlp_dim if self.mlp_dim is not None else int(self.hidden_dim * self.mlp_ratio)

        modulation = AdaLNModulation(
            hidden_dim=self.hidden_dim,
            cond_dim=self.cond_dim,
            use_gate=True,
            act_fn=self.act_fn,
        )

        if self.use_attention:
            scale_msa, shift_msa, gate_msa = modulation(conditioning)

            h = nn.LayerNorm()(x)
            if h.ndim == 2:
                h = h * (1.0 + scale_msa) + shift_msa
            else:
                scale_msa = jnp.expand_dims(scale_msa, 1)
                shift_msa = jnp.expand_dims(shift_msa, 1)
                gate_msa = jnp.expand_dims(gate_msa, 1)
                h = h * (1.0 + scale_msa) + shift_msa

            h = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=qkv_features,
                dropout_rate=self.dropout_rate,
            )(h, mask=mask, deterministic=not training)

            h = nn.Dropout(self.dropout_rate)(h, deterministic=not training)
            x = x + gate_msa * h

        scale_mlp, shift_mlp, gate_mlp = modulation(conditioning)

        h = nn.LayerNorm()(x)
        if h.ndim == 2:
            h = h * (1.0 + scale_mlp) + shift_mlp
        else:
            scale_mlp = jnp.expand_dims(scale_mlp, 1)
            shift_mlp = jnp.expand_dims(shift_mlp, 1)
            gate_mlp = jnp.expand_dims(gate_mlp, 1)
            h = h * (1.0 + scale_mlp) + shift_mlp

        h = nn.Dense(mlp_hidden_dim)(h)
        h = self.act_fn(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not training)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not training)

        x = x + gate_mlp * h

        return x


def _get_layers(
    layers: Layers_t,
    output_dim: int | None = None,
    dropout_rate: float | None = None,
) -> list[nn.Module]:
    """Get modules from layer parameters."""
    modules = []
    if isinstance(layers, Sequence):
        for layer in layers:
            layer = dict(layer)
            layer_type = layer.pop("layer_type", "mlp")
            if layer_type == "mlp":
                lay = MLPBlock(**layer)
            elif layer_type == "self_attention":
                lay = SelfAttentionBlock(**layer)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            modules.append(lay)
    if output_dim is not None:
        modules.append(nn.Dense(output_dim))
        if dropout_rate is not None:
            modules.append(nn.Dropout(dropout_rate))
    return modules


def _apply_modules(
    modules: list[nn.Module],
    conditions: jax.Array,
    attention_mask: jnp.ndarray | None,
    training: bool,
) -> jnp.ndarray:
    """Apply modules to conditions."""
    for module in modules:
        if isinstance(module, SelfAttentionBlock):
            conditions = module(conditions, attention_mask, training)
        elif isinstance(module, nn.Dense):
            conditions = module(conditions)
        elif isinstance(module, nn.Dropout):
            conditions = module(conditions, deterministic=not training)
        else:
            conditions = module(conditions, training)
    return conditions




