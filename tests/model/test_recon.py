"""Tests for the gene-expression autoencoder (``scaleflow.model._recon``)."""

import jax
import jax.numpy as jnp
import numpy as np

from scaleflow.model._recon import (
    Autoencoder,
    Decoder,
    Encoder,
    reconstruction_loss,
)

N_CELLS = 256
N_GENES = 50
LATENT = 8
PRED_LATENT = 16  # dimensionality of a "predefined" latent for the decoder-only test


def _low_rank_genes(seed: int = 0) -> np.ndarray:
    """Non-negative, low-rank synthetic expression an AE can actually compress."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((N_CELLS, LATENT)).astype(np.float32)
    w = rng.standard_normal((LATENT, N_GENES)).astype(np.float32)
    return np.maximum(z @ w, 0.0).astype(np.float32)


class TestEncoderDecoderShapes:
    def test_encode_shape(self):
        x = jnp.asarray(_low_rank_genes())
        enc = Encoder(latent_dim=LATENT, hidden_dims=(32, 16))
        params = enc.init(jax.random.PRNGKey(0), x, training=False)["params"]
        z = enc.apply({"params": params}, x, training=False)
        assert z.shape == (N_CELLS, LATENT)

    def test_decode_shape(self):
        z = jnp.ones((N_CELLS, LATENT))
        dec = Decoder(output_dim=N_GENES, hidden_dims=(16, 32))
        params = dec.init(jax.random.PRNGKey(0), z, training=False)["params"]
        x_hat = dec.apply({"params": params}, z, training=False)
        assert x_hat.shape == (N_CELLS, N_GENES)

    def test_encoder_no_hidden_is_linear(self):
        x = jnp.asarray(_low_rank_genes())
        enc = Encoder(latent_dim=LATENT, hidden_dims=())
        params = enc.init(jax.random.PRNGKey(0), x, training=False)["params"]
        z = enc.apply({"params": params}, x, training=False)
        assert z.shape == (N_CELLS, LATENT)


class TestAutoencoder:
    def test_roundtrip_shapes_and_separable_methods(self):
        x = jnp.asarray(_low_rank_genes())
        ae = Autoencoder(
            gene_dim=N_GENES,
            latent_dim=LATENT,
            encoder_hidden=(32, 16),
            decoder_hidden=(16, 32),
        )
        params = ae.init(jax.random.PRNGKey(0), x, training=False)["params"]

        x_recon, z = ae.apply({"params": params}, x, training=False)
        assert x_recon.shape == (N_CELLS, N_GENES)
        assert z.shape == (N_CELLS, LATENT)

        # encode / decode applied separately (string-method form) must match __call__.
        z2 = ae.apply({"params": params}, x, training=False, method="encode")
        x_hat = ae.apply({"params": params}, z2, training=False, method="decode")
        assert z2.shape == (N_CELLS, LATENT)
        assert x_hat.shape == (N_CELLS, N_GENES)
        np.testing.assert_allclose(np.asarray(z), np.asarray(z2), atol=1e-5)
        np.testing.assert_allclose(np.asarray(x_recon), np.asarray(x_hat), atol=1e-5)


class TestFit:
    def test_ae_fit_loss_decreases(self):
        x = _low_rank_genes(seed=1)
        ae = Autoencoder(
            gene_dim=N_GENES,
            latent_dim=LATENT,
            encoder_hidden=(32,),
            decoder_hidden=(32,),
        )
        _, losses = ae.train(x, n_iters=300, batch_size=64, lr=1e-2, seed=0)
        assert len(losses) == 300
        assert losses[-1] < losses[0]

    def test_decoder_only_predefined_latent_fit(self):
        # Genes are a linear function of a predefined latent Z -> the decoder must learn it.
        rng = np.random.default_rng(2)
        Z = rng.standard_normal((N_CELLS, PRED_LATENT)).astype(np.float32)
        W = rng.standard_normal((PRED_LATENT, N_GENES)).astype(np.float32)
        X = (Z @ W).astype(np.float32)
        dec = Decoder(output_dim=N_GENES, hidden_dims=(32,))
        _, losses = dec.train(Z, X, n_iters=300, batch_size=64, lr=1e-2, seed=0)
        assert losses[-1] < losses[0]


class TestLoss:
    def test_mse_zero_on_identity(self):
        x = jnp.asarray(_low_rank_genes())
        assert float(reconstruction_loss(x, x, kind="mse")) == 0.0

    def test_mse_positive_on_mismatch(self):
        x = jnp.asarray(_low_rank_genes(seed=3))
        y = jnp.asarray(_low_rank_genes(seed=4))
        assert float(reconstruction_loss(x, y, kind="mse")) > 0.0


if __name__ == "__main__":
    # Allow running without pytest (the `pancellflow` env has no pytest):
    #   PYTHONPATH=<sub>/src python tests/model/test_recon.py
    # Still discovered normally under pytest.
    import sys
    import traceback

    import scaleflow.model._recon as _recon_mod

    print(f"scaleflow.model._recon -> {_recon_mod.__file__}")
    failures = 0
    for _cls_name, _cls in sorted(globals().items()):
        if _cls_name.startswith("Test") and isinstance(_cls, type):
            _inst = _cls()
            for _m in sorted(dir(_inst)):
                if _m.startswith("test_"):
                    try:
                        getattr(_inst, _m)()
                        print(f"PASS {_cls_name}.{_m}")
                    except Exception:  # noqa: BLE001
                        failures += 1
                        print(f"FAIL {_cls_name}.{_m}")
                        traceback.print_exc()
    print(f"\n{'OK' if failures == 0 else 'FAILED'}: {failures} failure(s)")
    sys.exit(1 if failures else 0)
