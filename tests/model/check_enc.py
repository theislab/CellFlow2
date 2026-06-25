# scripts/check_enc.py  — run inside the training container (jax env)
import numpy as np
from functools import partial
import jax
from jax.tree_util import tree_flatten_with_path, keystr
from scaleflow.model import ScaleFlow
from scaleflow.utils import match_linear

COND_SHAPES = {"cell_line": 300, "drug": 256, "prophet": 512, "dose": 1}
ALL_MODS    = ["cell_line", "drug", "prophet", "dose"]

def build(keys):
    B = 64
    batch = {
        "src_cell_data": np.zeros((B, 2058), np.float32),
        "tgt_cell_data": np.zeros((B, 2058), np.float32),
        "condition": {k: np.zeros((1, 1, COND_SHAPES[k]), np.float32) for k in keys},
    }
    enc = [{"layer_type": "mlp", "dims": [1024, 1024], "dropout_rate": 0.0}]
    sf = ScaleFlow(solver="otfm")
    sf.prepare_model(
        sample_batch=batch, max_combination_length=1, conditioning="concatenation",
        pooling="attention_token",
        layers_before_pool={k: enc for k in keys},
        layers_after_pool=enc,
        cond_output_dropout=0.9,
        hidden_dims=(512,), decoder_dims=(512,), condition_embedding_dim=256,
        match_fn=partial(match_linear, epsilon=1.0),
    )
    params = sf.solver.vf_state.params
    paths  = [keystr(p) for p, _ in tree_flatten_with_path(params)[0]]
    n      = int(sum(x.size for x in jax.tree.leaves(params)))
    return paths, n

def report(label, keys):
    paths, n = build(keys)
    print(f"=== {label}: {keys} ===")
    for mod in ALL_MODS:
        hits = [pp for pp in paths if mod in pp]
        mark = "✓ has encoder" if hits else ("—" if mod in keys else "✗ absent (as expected)")
        print(f"  {mod:10}: {len(hits):2d} arrays   {mark}")
    print(f"  total params: {n:,}\n")
    return n

n_prophet = report("prophet mode", ["cell_line", "drug", "prophet", "dose"])
n_default = report("default mode", ["cell_line", "drug", "dose"])

print(f"prophet encoder contributes ~{n_prophet - n_default:,} params "
      f"({100*(n_prophet-n_default)/n_default:.1f}% more than default)")
