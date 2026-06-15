"""
Self-contained correctness test for temp_edit.py's diagnostic functions.

Run:  python experiments/test_temp_edit.py

`temp_edit` imports scaleflow's metric functions; this test uses the real ones when
they import, else installs reference stubs matching their documented semantics
(`compute_r_squared(x,y) == r2_score(mean(x,0), mean(y,0))`; an MMD with mmd(a,a)=0).
All assertions hold for EITHER backend:
  • effect sizes / effect_ratio / scalar aggregations are metric-independent (exact),
  • gap_closure / r_squared(_delta) are checked on perfect-prediction and predict-control
    cases, whose values (1 and 0) are invariants of any valid metric.
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── backend: real scaleflow metrics if importable, else reference stubs ───────
def _install_stubs():
    def _r2(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - a.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    m = types.ModuleType("scaleflow.metrics._metrics")
    m.compute_r_squared = lambda x, y: _r2(np.mean(np.asarray(x, float), 0), np.mean(np.asarray(y, float), 0))
    m.compute_scalar_mmd = lambda x, y: float(np.sum((np.mean(np.asarray(x, float), 0) - np.mean(np.asarray(y, float), 0)) ** 2))
    m.compute_e_distance_fast = lambda x, y: float(np.linalg.norm(np.mean(np.asarray(x, float), 0) - np.mean(np.asarray(y, float), 0)))

    cb = types.ModuleType("scaleflow.training._callbacks")
    class ComputationCallback:  # minimal base
        pass
    cb.ComputationCallback = ComputationCallback

    for n in ("scaleflow", "scaleflow.metrics", "scaleflow.training"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules["scaleflow.metrics._metrics"] = m
    sys.modules["scaleflow.training._callbacks"] = cb


try:
    import scaleflow.metrics._metrics            # noqa: F401
    import scaleflow.training._callbacks         # noqa: F401
    BACKEND = "real scaleflow"
except Exception:
    _install_stubs()
    BACKEND = "reference stub (self-contained)"

import temp_edit as te  # noqa: E402


# ── tiny assert harness ──────────────────────────────────────────────────────
_FAILS = []
def check(name, got, want, atol=1e-6):
    ok = np.isclose(got, want, atol=atol, rtol=0)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got:.6g}  want={want:.6g}")
    if not ok:
        _FAILS.append(name)

def check_true(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        _FAILS.append(name)


def const(mean_vec, n=64):
    """n cells all equal to mean_vec → exact mean, with a tiny deterministic jitter."""
    return np.tile(np.asarray(mean_vec, float), (n, 1))


# ── tests ────────────────────────────────────────────────────────────────────
def test_effect_sizes():
    print("\n# effect sizes (metric-independent, exact)")
    src  = const([0.0, 0.0])          # control mean (0,0)
    true = const([3.0, 4.0])          # ‖(3,4)-(0,0)‖ = 5
    pred = const([1.5, 2.0])          # ‖(1.5,2)-(0,0)‖ = 2.5
    df = te.condition_diagnostics({"c": src}, {"c": true}, {"c": pred})
    r = df.iloc[0]
    check("true_effect", r.true_effect, 5.0)
    check("pred_effect", r.pred_effect, 2.5)
    check("effect_ratio", r.effect_ratio, 0.5)


def test_perfect_prediction():
    print("\n# perfect prediction  pred == true")
    rng = np.random.default_rng(0)
    src  = rng.normal(0.0, 0.5, size=(80, 3))
    true = rng.normal(0.0, 0.5, size=(80, 3)) + np.array([2.0, -1.0, 3.0])
    pred = true.copy()
    df = te.condition_diagnostics({"c": src}, {"c": true}, {"c": pred})
    r = df.iloc[0]
    check("r_squared (=1)", r.r_squared, 1.0, atol=1e-6)
    check("r_squared_delta (=1)", r.r_squared_delta, 1.0, atol=1e-6)
    check("effect_ratio (=1)", r.effect_ratio, 1.0, atol=1e-6)
    check_true("gap_closure ≈ 1 (>0.99)", r.gap_closure > 0.99)


def test_predict_control():
    print("\n# degenerate: predict the control  pred == source")
    rng = np.random.default_rng(1)
    src  = rng.normal(0.0, 0.5, size=(80, 3))
    true = rng.normal(0.0, 0.5, size=(80, 3)) + np.array([2.0, -1.0, 3.0])
    pred = src.copy()
    df = te.condition_diagnostics({"c": src}, {"c": true}, {"c": pred})
    r = df.iloc[0]
    check("pred_effect (=0)", r.pred_effect, 0.0, atol=1e-6)
    check("effect_ratio (=0)", r.effect_ratio, 0.0, atol=1e-6)
    check("gap_closure (=0)", r.gap_closure, 0.0, atol=1e-6)


def test_scalar_diagnostics():
    print("\n# scalar_diagnostics aggregation (exact, metric-independent)")
    import pandas as pd
    df = pd.DataFrame({
        "true_effect":  [1.0, 2.0, 3.0, 4.0],
        "pred_effect":  [0.5, 1.0, 1.5, 2.0],   # exactly 0.5 * true → slope 0.5, corr 1
        "effect_ratio": [0.5, 0.5, 0.5, 0.5],
        "gap_closure":  [0.5, -0.2, 0.8, 0.1],  # mean 0.3, 3/4 positive
        "r_squared":    [0.9, 0.8, 0.95, 0.7],  # mean 0.8375
    })
    s = te.scalar_diagnostics(df)
    check("gap_closure_mean", s["gap_closure_mean"], 0.3)
    check("gap_closure_frac_pos", s["gap_closure_frac_pos"], 0.75)
    check("effect_calib_slope", s["effect_calib_slope"], 0.5)
    check("effect_corr", s["effect_corr"], 1.0)
    check("effect_ratio_mean", s["effect_ratio_mean"], 0.5)
    check("r_squared_mean", s["r_squared_mean"], 0.8375)


def test_subsample_and_shape():
    print("\n# _subsample + per-condition table shape")
    rng = np.random.default_rng(2)
    a = np.arange(20).reshape(20, 1).astype(float)
    sub = te._subsample(a, 5, rng)
    check_true("_subsample caps to n", sub.shape[0] == 5)
    check_true("_subsample subset of original", set(sub.ravel()).issubset(set(a.ravel())))
    check_true("_subsample no-op when n>=len", te._subsample(a, 99, rng).shape[0] == 20)

    src = {"c0": const([0, 0]), "c1": const([0, 0])}
    true = {"c0": const([1, 0]), "c1": const([0, 2])}
    pred = {"c0": const([1, 0]), "c1": const([0, 2])}
    df = te.condition_diagnostics(src, true, pred)
    check_true("one row per condition", len(df) == 2)
    check_true("has expected columns",
               {"condition", "true_effect", "pred_effect", "gap_closure", "r_squared"} <= set(df.columns))


def test_over_datasets():
    print("\n# diagnostics_over_datasets (nested dict → split/dataset rows)")
    src  = {"ds": {"c": const([0, 0])}}
    true = {"ds": {"c": const([3, 4])}}
    pred = {"ds": {"c": const([3, 4])}}
    df = te.diagnostics_over_datasets(src, true, pred)
    check_true("dataset column present", "dataset" in df.columns and df.iloc[0].dataset == "ds")
    check("true_effect across datasets", df.iloc[0].true_effect, 5.0)


def main():
    print(f"backend: {BACKEND}")
    test_effect_sizes()
    test_perfect_prediction()
    test_predict_control()
    test_scalar_diagnostics()
    test_subsample_and_shape()
    test_over_datasets()
    print("\n" + ("ALL TESTS PASSED" if not _FAILS else f"FAILURES: {_FAILS}"))
    sys.exit(1 if _FAILS else 0)


if __name__ == "__main__":
    main()
