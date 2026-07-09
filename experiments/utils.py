"""Prophet-ablation condition transform and the optimizer builder."""
from __future__ import annotations

import numpy as np
import optax
from omegaconf import DictConfig


class ConditionTransform:
    """Rewrites the 'prophet' condition at sample time.

    default: drop 'prophet'.  prophet: keep as-is (caller passes None instead).
    random:  replace with random vectors — fresh each train step, fixed per
             condition at validation (keyed by cond_key).
    """

    def __init__(self, mode: str, seed: int = 42):
        self.mode  = mode
        self._rng  = np.random.default_rng(seed)
        self._seed = seed
        self._cache: dict = {}

    def __call__(self, cond: dict, cond_key: str | None = None) -> dict:
        if self.mode == "prophet":
            return cond
        result = {}
        for k, v in cond.items():
            if k != "prophet":
                result[k] = v
            elif self.mode == "random":
                if cond_key is not None:
                    cache_key = (cond_key, k, v.shape)
                    if cache_key not in self._cache:
                        s = abs(hash(cond_key + k + str(self._seed))) % (2 ** 31)
                        self._cache[cache_key] = np.random.default_rng(s).standard_normal(v.shape).astype(v.dtype)
                    result[k] = self._cache[cache_key]
                else:
                    result[k] = self._rng.standard_normal(v.shape).astype(v.dtype)
        return result


def build_optimizer(cfg: DictConfig):
    """Warmup-cosine Adam wrapped in MultiSteps for gradient accumulation.

    Schedule is in optimizer-update steps (iterations / accumulation).
    Returns (optimizer, schedule).
    """
    t     = cfg.training
    accum = int(t.get("grad_accumulation", 1)) or 1
    steps = max(int(t.num_iterations) // accum, 1)
    warmup = max(min(int(t.warmup_iterations) // accum, steps - 1), 1)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=float(t.get("init_lr", 0.0)),
        peak_value=float(t.peak_lr),
        warmup_steps=warmup,
        decay_steps=steps,
        end_value=float(t.end_lr),
    )
    opt = optax.adam(schedule)
    return (optax.MultiSteps(opt, accum) if accum > 1 else opt), schedule


def _norm_key(c):
    """Condition key as a clean tuple (parses the 1-tuple-holding-str(tuple) form)."""
    import ast
    if isinstance(c, (tuple, list)) and len(c) == 1 and isinstance(c[0], str) and c[0].lstrip().startswith("("):
        try:
            return tuple(ast.literal_eval(c[0]))
        except (ValueError, SyntaxError):
            pass
    return tuple(c) if isinstance(c, (tuple, list)) else (c,)


def _cond_label(c) -> str:
    return "·".join(str(x) for x in _norm_key(c))


def umap_effect_panels(
    src: dict, true: dict, preds: dict, background: dict | None = None,
    k: int = 6, decoder=None, seed: int = 0, out_path: str | None = None,
    min_cells: int = 20, max_cells: int = 400, bg_max: int = 1500, title: str = "model comparison",
):
    """Qualitative per-condition UMAP overlay on the highest-effect conditions.

    For each of the top-``k`` conditions (ranked by true effect ‖mean(true) − mean(ctrl)‖), fits a
    UMAP on: a light-grey backdrop of the cell line's OTHER perturbations (``background``, heavily
    subsampled — anchors the manifold so the perturbation SIZE is visible), plus this condition's
    control (grey), true-perturbed (blue), and each model's predicted-perturbed cells (a distinct
    colour each). Shows whether a model's predicted cloud lands on the true-perturbed manifold — and,
    with several models, which is closest (e.g. prophet vs no-prophet) and whether it under-shoots.
    Panel title reports the true effect + each model's effect-ratio ‖pred−ctrl‖/‖true−ctrl‖ (<1 ⇒ under-shoot).

    Parameters
    ----------
    src, true : {cond_key: (n, d)}
        Control/source and true-perturbed cells per condition, in the flow's latent space.
    preds : {model_label: {cond_key: (n, d)}}
        Predicted-perturbed cells per model (same cond_keys as ``true``).
    background : {cell_line: (N, d)} | None
        Pool of the cell line's cells across all perturbations (light-grey manifold context). The cell
        line is taken from the first element of the (normalised) cond_key.
    decoder : ReconDecoder | None
        If given, decode latents → genes and UMAP in gene space; else UMAP the raw latent.
    Returns the effect-ranking :class:`pandas.DataFrame`; saves a (k×1) figure to ``out_path``.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import umap

    rng = np.random.default_rng(seed)
    labels = list(preds)

    def _sub(a, n):
        a = np.asarray(a, np.float32)
        return a[rng.choice(a.shape[0], n, replace=False)] if a.shape[0] > n else a

    def _space(a):
        return np.asarray(decoder.decode(np.asarray(a, np.float32)), np.float32) if decoder is not None else np.asarray(a, np.float32)

    # ── rank conditions by true effect; require enough cells + presence in every model ──
    rows = []
    for c in true:
        if c not in src or true[c].shape[0] < min_cells or src[c].shape[0] < min_cells:
            continue
        if not all(c in preds[m] for m in labels):
            continue
        cm = np.asarray(src[c], np.float32).mean(0)
        eff = float(np.linalg.norm(np.asarray(true[c], np.float32).mean(0) - cm))
        row = {"cond": c, "true_effect": eff}
        for m in labels:
            pe = float(np.linalg.norm(np.asarray(preds[m][c], np.float32).mean(0) - cm))
            row[f"ratio_{m}"] = pe / (eff + 1e-8)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("true_effect", ascending=False).reset_index(drop=True)
    top = df.head(k)
    if len(top) == 0:
        raise ValueError("No conditions passed the min_cells / presence filter.")

    palette = ["#B12F8C", "#E1812C", "#3A923A", "#9467BD", "#8C564B"]
    mcol = {m: palette[i % len(palette)] for i, m in enumerate(labels)}

    fig, axes = plt.subplots(len(top), 1, figsize=(7, 5 * len(top)))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, top.iterrows()):
        c = r["cond"]
        cl = str(_norm_key(c)[0])
        # backdrop first (fit + drawn behind) so control/true/pred sit inside the cell line's manifold
        groups = []
        if background is not None and cl in background:
            groups.append(("other perturbations", _space(_sub(background[cl], bg_max)), "#DDDDDD", 3, 0.25))
        groups += [("control", _space(_sub(src[c], max_cells)), "#8A8A8A", 7, 0.55),
                   ("true",    _space(_sub(true[c], max_cells)), "#3F8AA6", 12, 0.9)]
        for m in labels:
            groups.append((m, _space(_sub(preds[m][c], max_cells)), mcol[m], 12, 0.9))
        X = np.concatenate([g[1] for g in groups], 0)
        emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed).fit_transform(X)
        i0 = 0
        for name, arr, col, sz, al in groups:   # drawn in order → backdrop behind, preds on top
            e = emb[i0:i0 + arr.shape[0]]; i0 += arr.shape[0]
            ax.scatter(e[:, 0], e[:, 1], s=sz, c=col, alpha=al, label=name, linewidths=0)
        rt = "  ".join(f"{m}={r[f'ratio_{m}']:.2f}" for m in labels)
        ax.set_title(f"{_cond_label(c)}   true_eff={r['true_effect']:.2f}   effect-ratio[{rt}]", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=7, loc="best", framealpha=0.6)
    fig.suptitle(title, y=1.002, fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"  saved UMAP → {out_path}")
    return df
