"""Minimal smoke tests for the ScaleFlow model.

ScaleFlow subclasses :class:`cellflow.model.CellFlow`, so data preparation, training,
and prediction are inherited from cellflow; scaleflow overrides only the velocity-field
and solver construction (``prepare_model``). This exercises that inherited data path
(``prepare_data`` → cellflow ``TrainSampler`` → ``prepare_model`` → ``train``) end to end.
"""

import numpy as np
import pytest
from cellflow.data import TrainSampler

from scaleflow.model import ScaleFlow


class TestScaleFlowSmoke:
    @pytest.mark.parametrize("solver", ["sf_otfm", "sf_genot"])
    def test_prepare_and_train(self, adata_perturbation, solver):
        sf = ScaleFlow(solver=solver)
        sf.prepare_data(
            adata=adata_perturbation,
            sample_rep="X",
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
            perturbation_covariate_reps={"drug": "drug"},
        )
        # scaleflow's `prepare_model` needs a sample batch (for the data dim + a sample
        # condition); build one from cellflow's TrainSampler on the inherited train_data.
        sample_batch = TrainSampler(sf.train_data, batch_size=8).sample(np.random.default_rng(0))

        vf_kwargs = {"genot_source_dims": (8, 8), "genot_source_dropout": 0.1} if solver == "sf_genot" else None
        sf.prepare_model(
            sample_batch=sample_batch,
            max_combination_length=1,
            pooling="mean",
            condition_embedding_dim=8,
            time_freqs=8,
            time_encoder_dims=(8, 8),
            hidden_dims=(8, 8),
            decoder_dims=(8, 8),
            vf_kwargs=vf_kwargs,
        )
        assert sf.solver is not None

        # `train` is inherited from cellflow: it builds the sampler from `train_data` and
        # drives scaleflow's CFG-aware trainer (created in `prepare_model`).
        sf.train(num_iterations=2, valid_freq=100)  # valid_freq > num_iterations: skip validation
        assert sf.trainer is not None
        assert len(sf.trainer.training_logs["loss"]) > 0
