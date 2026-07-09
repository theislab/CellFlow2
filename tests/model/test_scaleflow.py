"""Minimal smoke tests for the ScaleFlow model.

NOTE: this is intentionally small coverage. The previous test_scaleflow.py tested
the now-removed dead ``CellFlow`` class against the old DataManager API. Fuller
ScaleFlow coverage (predict, validation, serialization, genot/eqm) is tracked
separately.
"""

import numpy as np
import pytest

from scaleflow.data._dataloader import InMemorySampler
from scaleflow.model import ScaleFlow


class TestScaleFlowSmoke:
    @pytest.mark.parametrize("solver", ["otfm", "genot"])
    def test_prepare_and_train(self, sample_grouped_distribution, solver):
        sampler = InMemorySampler(
            data=sample_grouped_distribution,
            rng=np.random.default_rng(0),
            batch_size=8,
        )
        sampler.init_sampler()
        sample_batch = sampler.sample()
        assert "condition" in sample_batch

        sf = ScaleFlow(solver=solver)
        sf.prepare_model(
            sample_batch=sample_batch,
            max_combination_length=1,
            pooling="mean",
            condition_embedding_dim=8,
            time_freqs=8,
            time_encoder_dims=(8, 8),
            hidden_dims=(8, 8),
            decoder_dims=(8, 8),
        )
        assert sf.solver is not None

        sf.train(
            train_dataloader=sampler,
            num_iterations=2,
            valid_freq=100,  # > num_iterations: skip validation in the smoke test
        )
        assert sf.trainer is not None
        assert len(sf.trainer.training_logs["loss"]) > 0
