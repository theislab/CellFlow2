"""Qualitative check: AE_10_pretrained on Tahoe A549.

UMAPs in 3 spaces: 1) latent (10D), 2) decoded gene space, 3) gene PCA-50 space.
Saves plots + e-distance CSV to OUT_DIR.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import gc
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anndata.io import read_elem
from scipy.spatial.distance import cdist

from scaleflow.model import ScaleFlow
from scaleflow.data import AnnDataLocation, DataManager
from scaleflow.data._dataloader import ValidationSampler

# ── Config ────────────────────────────────────────────────────────────────
CKPT = '/lustre/groups/ml01/workspace/weixu.wang/panflow_cfp/CellFlow2/experiments/outputs/tahoe_pretrained_ae10_deterministic/ScaleFlow.pkl'
DECODER_CKPT = '/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/weights/tahoe/split03/AE/Default/400_[4096, 4096, 4096, 4096]_10_20251107/epoch=62-val/loss_epoch=0.0494.ckpt'
DATA_PATH = '/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert/tahoe_a549_w_emb.h5ad'
OUT_DIR = '/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/experiments/qualcheck_ae10_plots'

OBSM_KEY = 'AE_10_pretrained'
PALETTE = {'ctrl': '#4878CF', 'true': '#D65F5F', 'pred': '#6ACC65'}
N_FOCUS = 5
UMAP_MAX = 200
CTRL_CAP = 200
MAX_CELLS_PER_GROUP = 200

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────
def subsample(arr, n, rng):
    return arr if len(arr) <= n else arr[rng.choice(len(arr), n, replace=False)]


def spotlight_umap(ctrl, true, pred, drug, title, ax):
    rng = np.random.default_rng(42)
    parts, labels = [], []
    for arr, lab, cap in [(ctrl, 'ctrl', CTRL_CAP), (true, 'true', UMAP_MAX), (pred, 'pred', UMAP_MAX)]:
        s = subsample(arr, cap, rng).astype(np.float32)
        parts.append(s)
        labels.extend([lab] * len(s))

    X = np.vstack(parts)
    au = ad.AnnData(X=X, obs=pd.DataFrame({'source': labels}))
    au.obs_names = [str(i) for i in range(len(au))]
    n_pcs = min(50, au.n_obs - 1, au.n_vars - 1)
    sc.pp.pca(au, n_comps=n_pcs)
    sc.pp.neighbors(au, n_pcs=n_pcs)
    sc.tl.umap(au)
    co = au.obsm['X_umap']

    for lab in ['ctrl', 'true', 'pred']:
        m = (au.obs['source'] == lab).values
        ax.scatter(co[m, 0], co[m, 1], c=PALETTE[lab], s=4, alpha=0.4,
                   label=f'{lab} [{m.sum()}]', rasterized=True)
        mu = co[m].mean(axis=0)
        ax.scatter(mu[0], mu[1], c=PALETTE[lab], marker='*', s=200,
                   edgecolors='k', linewidths=0.5, zorder=10)

    ax.set_title(f'{drug}\n{title}', fontsize=10)
    ax.axis('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.legend(fontsize=7, markerscale=2)
    del au
    gc.collect()


def e_dist(X, Y, n_max=500):
    rl = np.random.default_rng(42)
    if len(X) > n_max:
        X = X[rl.choice(len(X), n_max, replace=False)]
    if len(Y) > n_max:
        Y = Y[rl.choice(len(Y), n_max, replace=False)]
    return 2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print(f'Python: {sys.executable}', flush=True)
    import jax; print(f'jax={jax.__version__}', flush=True)
    import jaxlib; print(f'jaxlib={jaxlib.__version__}', flush=True)
    print(f'jax devices: {jax.devices()}', flush=True)

    # 1. Load model
    print('Loading ScaleFlow model...', flush=True)
    sf = ScaleFlow.load(CKPT)
    print('Model loaded.', flush=True)
    if not sf._solver.is_trained:
        sf._solver.is_trained = True
    print(f'  data_dim: {sf._data_dim}')

    # 2. Load data (subsampled)
    print('Reading obs...')
    with h5py.File(DATA_PATH, 'r') as f:
        obs = read_elem(f['obs'])

    obs['control'] = (obs['drug_0'] == 'control') & (obs['drug_1'] == 'control')

    rng = np.random.default_rng(42)
    keep_positions = []
    for drug, group in obs.groupby('drug_0', observed=True):
        positions = np.where(obs['drug_0'] == drug)[0]
        if len(positions) > MAX_CELLS_PER_GROUP:
            positions = rng.choice(positions, MAX_CELLS_PER_GROUP, replace=False)
        keep_positions.extend(positions.tolist())
    keep_positions = sorted(keep_positions)

    drug_counts = obs.loc[~obs['control']].groupby('drug_0', observed=True).size().sort_values(ascending=False)
    focus_drugs = drug_counts.head(N_FOCUS).index.tolist()
    print(f'Focus drugs: {focus_drugs}')
    print(f'Using {len(keep_positions):,} / {len(obs):,} cells')

    print('Extracting embeddings...')
    with h5py.File(DATA_PATH, 'r') as f:
        emb = np.array(f['obsm'][OBSM_KEY])[keep_positions]

    sub_obs = obs.iloc[keep_positions].copy()
    sub_obs.index = [str(i) for i in range(len(sub_obs))]

    adata = ad.AnnData(X=emb, obs=sub_obs)
    adata.obsm[OBSM_KEY] = emb

    full_adata = ad.read_h5ad(DATA_PATH, backed='r')
    for k in full_adata.uns:
        if 'embedding' in k or 'dim' in k:
            adata.uns[k] = full_adata.uns[k]
    del full_adata, obs
    adata.obs['control'] = (adata.obs['drug_0'] == 'control') & (adata.obs['drug_1'] == 'control')
    print(f'  adata: {adata.n_obs:,} cells')

    # 3. Predict
    print('Setting up DataManager...')
    adl = AnnDataLocation()
    data_manager = DataManager(
        dist_flag_key='control',
        src_dist_keys=['cell_line'],
        tgt_dist_keys=['drug_0'],
        rep_keys={
            'cell_line': 'cell_line_ccle_embeddings',
            'drug_0': 'drug_0_embeddings',
        },
        data_location=adl.obsm[OBSM_KEY],
    )

    gd = data_manager.prepare_data(adata)
    sampler = ValidationSampler(gd, n_conditions_on_log_iteration=None,
                                n_conditions_on_train_end=None, seed=42)
    if not sampler.initialized:
        sampler.init_sampler()

    annotation = sampler._data.annotation
    tgt_labels = annotation.tgt_dist_idx_to_labels
    tgt_to_src = sampler._tgt_to_src
    gd_data = sampler._data.data

    print('Predicting...')
    ctrl_emb, true_dict, pred_dict = None, {}, {}
    for tgt_idx in gd_data.conditions:
        src_idx = tgt_to_src.get(tgt_idx)
        if src_idx is None:
            continue
        drug = tgt_labels[tgt_idx][0] if tgt_idx in tgt_labels else str(tgt_idx)
        src = np.array(gd_data.src_data[src_idx])
        tgt = np.array(gd_data.tgt_data[tgt_idx])
        pred = np.array(sf._solver.predict(src, gd_data.conditions[tgt_idx]))
        if ctrl_emb is None:
            ctrl_emb = src
        true_dict[drug] = tgt
        pred_dict[drug] = pred
        print(f'  {drug}: src={src.shape[0]}, tgt={tgt.shape[0]}, pred={pred.shape[0]}')

    print(f'Predicted {len(pred_dict)} drugs')

    # 4. Load decoder
    print('Loading AE decoder...')
    import torch
    sys.path.insert(0, '/lustre/groups/ml01/workspace/xiaotong.fu/reconstruction/reconstruction/src')
    from sc_reconstruction.adapters.e2e_decoder_adapters import FrozenAEDecoderAdapter

    decoder = FrozenAEDecoderAdapter(DECODER_CKPT, map_location='cpu')
    decoder.eval()
    print(f'  Decoder: {decoder.n_input} -> {decoder.n_output}')

    @torch.no_grad()
    def decode(z):
        return decoder(torch.as_tensor(np.asarray(z, dtype=np.float32))).numpy()

    # 5. Plot: Latent / Gene / Gene-PCA for focus drugs
    print('Generating UMAPs...')
    for drug in focus_drugs:
        if drug not in pred_dict:
            print(f'  Skipping {drug}')
            continue

        ctrl_lat = ctrl_emb
        true_lat = true_dict[drug]
        pred_lat = pred_dict[drug]

        ctrl_gene = decode(ctrl_lat)
        true_gene = decode(true_lat)
        pred_gene = decode(pred_lat)

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        spotlight_umap(ctrl_lat, true_lat, pred_lat, drug, 'Latent space', axes[0])
        spotlight_umap(ctrl_gene, true_gene, pred_gene, drug, 'Gene space (decoded)', axes[1])

        all_gene = np.vstack([ctrl_gene, true_gene, pred_gene])
        mean = all_gene.mean(axis=0)
        _, _, vt = np.linalg.svd(all_gene - mean, full_matrices=False)
        pc50 = vt[:50]
        to_pca = lambda x, m=mean, p=pc50: (x - m) @ p.T
        spotlight_umap(to_pca(ctrl_gene), to_pca(true_gene), to_pca(pred_gene),
                       drug, 'Gene PCA-50 space', axes[2])

        plt.suptitle(f'AE_10_pretrained | A549 | {drug}', fontsize=13, y=1.02)
        plt.tight_layout()
        fname = f'umap_{drug.replace(" ", "_").replace("/", "_")}.png'
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {fname}')
        del ctrl_gene, true_gene, pred_gene
        gc.collect()

    # 6. E-distance summary
    print('Computing e-distances...')
    rows = []
    for drug in focus_drugs:
        if drug not in pred_dict:
            continue
        pred_gene = decode(pred_dict[drug])
        true_gene = decode(true_dict[drug])
        ctrl_gene = decode(ctrl_emb)
        rows.append({
            'drug': drug,
            'ed_pred_gene': e_dist(pred_gene, true_gene),
            'ed_ctrl_gene': e_dist(ctrl_gene, true_gene),
            'ed_pred_latent': e_dist(pred_dict[drug], true_dict[drug]),
            'ed_ctrl_latent': e_dist(ctrl_emb, true_dict[drug]),
        })

    df = pd.DataFrame(rows)
    df['gene_win'] = df['ed_pred_gene'] < df['ed_ctrl_gene']
    df['latent_win'] = df['ed_pred_latent'] < df['ed_ctrl_latent']
    df.to_csv(os.path.join(OUT_DIR, 'edistance.csv'), index=False)
    print(f'Gene win rate: {df["gene_win"].mean():.0%}')
    print(f'Latent win rate: {df["latent_win"].mean():.0%}')
    print(df.to_string())
    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()
