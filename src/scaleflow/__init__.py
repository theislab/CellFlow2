from importlib import metadata

# All submodules are loaded lazily to avoid the scanpy → numba → numpy≤1.26
# conflict at import time.  They are imported on first attribute access.
_LAZY_SUBMODULES = {
    "data",
    "datasets",
    "metrics",
    "model",
    "networks",
    "pp",
    "preprocessing",
    "solvers",
    "training",
    "utils",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        mod_name = "preprocessing" if name == "pp" else name
        mod = importlib.import_module(f"scaleflow.{mod_name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
