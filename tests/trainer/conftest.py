import jax.numpy as jnp
import pytest


@pytest.fixture
def dataloader():
    """Minimal training sampler: scaleflow samplers expose ``sample()`` (internal rng)."""

    class DataLoader:
        n_conditions = 10

        def sample(self):
            return {
                "src_cell_data": jnp.ones((10, 5)) * 10,
                "tgt_cell_data": jnp.ones((10, 5)),
                "condition": {"pert1": jnp.ones((1, 2, 3))},
            }

    return DataLoader()


@pytest.fixture
def valid_loader():
    """Minimal validation sampler in scaleflow's per-condition format.

    ``sample()`` returns ``{"source", "condition", "target"}``, each mapping a
    condition key -> data (see ScaleFlow's ValidationSampler / trainer._validation_step).
    """

    class ValidationSampler:
        def sample(self, mode="on_log_iteration"):
            return {
                "source": {"my_naming_of_pert": jnp.ones((10, 5)) * 10},
                "condition": {"my_naming_of_pert": {"pert1": jnp.ones((1, 2, 3))}},
                "target": {"my_naming_of_pert": jnp.ones((10, 5))},
            }

    return {"val": ValidationSampler()}
