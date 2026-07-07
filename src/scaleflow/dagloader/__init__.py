"""Declarative, index-free class-mapping sampler over annbatch.

A :class:`Scheme` is a rooted tree of :class:`Node`\\s over named cell sources; each node partitions
its source's cells into leaves (unique column-combinations) with a per-combination weight mapping.
:class:`DAGClassLoader` streams matched ``{source, target, condition}`` batches — one condition per
batch — where every node streams through its own :class:`ScheduledClassSampler`, the root's per-batch
category is drawn from its weights, and each bound child's category is *derived* from the parent's so
the loaders zip batch-for-batch. No row indices are exposed: the scheme is columns / keys / weights.

See ``README.md`` (next to this file) for the model, the sampling schemes, and the mapping to
cellflow and sc-flow-tools use-cases.
"""

from scaleflow.dagloader._loader import DAGClassLoader
from scaleflow.dagloader._schema import (
    Bind,
    Container,
    Node,
    SamplerConfig,
    Scheme,
    Weights,
    frequency,
    inverse_frequency,
    uniform,
)
from scaleflow.dagloader._scheduled_sampler import ScheduledClassSampler
from scaleflow.dagloader._schemes import perturbation_scheme

__all__ = [
    "Bind",
    "Container",
    "DAGClassLoader",
    "Node",
    "SamplerConfig",
    "Scheme",
    "ScheduledClassSampler",
    "Weights",
    "frequency",
    "inverse_frequency",
    "perturbation_scheme",
    "uniform",
]
