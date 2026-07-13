from collections.abc import Callable

import jax.numpy as jnp
from cellflow.networks._utils import BaseModule
from flax import linen as nn
from flax.linen import initializers

__all__ = [
    "AdaLNModulation",
    "AdaLNZeroBlock",
    "build_adaln_blocks",
    "apply_adaln",
]


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


def build_adaln_blocks(
    *,
    decoder_dims,
    cond_dim: int,
    decoder_dropout: float,
    act_fn: Callable[[jnp.ndarray], jnp.ndarray],
    conditioning_kwargs: dict,
) -> list["AdaLNZeroBlock"]:
    """Build the per-cell AdaLN-Zero block stack (one block per ``decoder_dims`` entry).

    Shared by the velocity fields' ``adaln_zero`` conditioning so the block construction
    lives in one place; each caller supplies its own ``cond_dim`` (the size of the
    conditioning vector it will modulate by: e.g. ``(t, cond)`` for OTFM, ``cond`` for
    EqM, ``(t, x_0, cond)`` for GENOT).
    """
    return [
        AdaLNZeroBlock(
            hidden_dim=dim,
            cond_dim=cond_dim,
            num_heads=conditioning_kwargs.get("num_heads", 8),
            qkv_dim=conditioning_kwargs.get("qkv_dim", None),
            mlp_ratio=conditioning_kwargs.get("mlp_ratio", 4.0),
            dropout_rate=decoder_dropout,
            use_attention=False,
            act_fn=act_fn,
        )
        for dim in decoder_dims
    ]


def apply_adaln(
    *,
    adaln_blocks,
    output_layer,
    out: jnp.ndarray,
    conditioning_vec: jnp.ndarray,
    squeeze: bool,
    train: bool,
) -> jnp.ndarray:
    """Run the AdaLN-Zero stack, modulating each cell (batch dim) by ``conditioning_vec``.

    ``out`` is the token to modulate (the encoded ``x``); ``conditioning_vec`` is the
    per-cell modulation signal assembled by the caller. No cross-cell attention — cells
    carry their own condition. Applies ``output_layer`` at the end.
    """
    if squeeze:
        out = jnp.expand_dims(out, 0)
        conditioning_vec = jnp.expand_dims(conditioning_vec, 0)
    for block in adaln_blocks:
        out = block(out, conditioning_vec, mask=None, training=train)
    if squeeze:
        out = jnp.squeeze(out, 0)
    return output_layer(out)
