# scaleflow's GENOT had diverged only by scaleflow-specific import sources, a
# lazy `jax.tree.map` predict return, and the absence of cellflow's `_match_kwargs`
# registry hook — all subsumed by cellflow's version — so it is re-exported rather
# than duplicated.
from cellflow.solvers._genot import GENOT

__all__ = ["GENOT"]
