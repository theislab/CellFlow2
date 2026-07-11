# scaleflow's ConditionEncoder was byte-identical to cellflow's apart from the
# pre-pool concatenation axis (scaleflow had a diverged ``axis=-2``; cellflow's
# ``axis=-1`` is correct now that condition dataloading is owned by cellflow), so
# it is re-exported rather than duplicated.
from cellflow.networks._set_encoders import ConditionEncoder

__all__ = ["ConditionEncoder"]
