# scaleflow's OTFlowMatching diverged from cellflow's only by its per-call
# classifier-free guidance (``predict(guidance_scale=w)`` + ``cfg_enabled``) used
# for the validation w-sweep. That was upstreamed into cellflow's OTFlowMatching
# (per-call ``guidance_scale`` composes with the construction-time ``guidance``), so
# the solver and its guidance strategies are now re-exported rather than duplicated.
from cellflow.solvers._otfm import ClassifierFreeGuidance, Guidance, OTFlowMatching

__all__ = ["OTFlowMatching", "ClassifierFreeGuidance", "Guidance"]
