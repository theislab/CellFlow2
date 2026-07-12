from cellflow.solvers import SOLVER_REGISTRY, register_solver
from cellflow.solvers._genot import GENOT
from cellflow.solvers._otfm import ClassifierFreeGuidance, Guidance, OTFlowMatching

from scaleflow.networks._velocity_field import (
    ConditionalVelocityField,
    EquilibriumVelocityField,
    GENOTConditionalVelocityField,
)
from scaleflow.solvers._eqm import EquilibriumMatching

# Register scaleflow's (solver, velocity-field) pairings into cellflow's shared registry under
# scaleflow-namespaced ``sf_*`` keys, so they do NOT override cellflow's own ``otfm``/``genot``
# entries. Each reuses cellflow's solver class with scaleflow's velocity-field variant (which adds
# ``cell_transformer``/``adaln_zero``); EqM is scaleflow-only. These are the ``solver=`` values
# the model accepts (e.g. ``ScaleFlow(solver="sf_otfm")``).
register_solver("sf_otfm", OTFlowMatching, ConditionalVelocityField)
register_solver("sf_genot", GENOT, GENOTConditionalVelocityField)
register_solver("sf_eqm", EquilibriumMatching, EquilibriumVelocityField)

__all__ = [
    "GENOT",
    "OTFlowMatching",
    "EquilibriumMatching",
    "ClassifierFreeGuidance",
    "Guidance",
    "SOLVER_REGISTRY",
    "register_solver",
]
