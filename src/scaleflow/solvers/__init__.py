from scaleflow.networks._velocity_field import (
    ConditionalVelocityField,
    EquilibriumVelocityField,
    GENOTConditionalVelocityField,
)
from scaleflow.solvers._eqm import EquilibriumMatching
from scaleflow.solvers._genot import GENOT
from scaleflow.solvers._otfm import ClassifierFreeGuidance, Guidance, OTFlowMatching
from scaleflow.solvers._registry import SOLVER_REGISTRY, register_solver

# Built-in solvers, resolved by name in `CellFlow(solver=...)`. OTFM/GENOT are cellflow's
# (re-exported) paired with scaleflow's velocity-field variants; EqM is scaleflow-only.
register_solver("otfm", OTFlowMatching, ConditionalVelocityField)
register_solver("genot", GENOT, GENOTConditionalVelocityField)
register_solver("eqm", EquilibriumMatching, EquilibriumVelocityField)

__all__ = [
    "GENOT",
    "OTFlowMatching",
    "EquilibriumMatching",
    "ClassifierFreeGuidance",
    "Guidance",
    "SOLVER_REGISTRY",
    "register_solver",
]
