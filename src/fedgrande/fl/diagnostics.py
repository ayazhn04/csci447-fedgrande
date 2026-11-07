"""Structural-consensus and update-disagreement diagnostics for GRANDE clients.

Every client shares the same ``features_by_estimator`` buffer, so the selected ORIGINAL (encoded) feature of internal
node (e, i) is ``features_by_estimator[e, argmax_v split_index_array[e, i, v]]`` -- directly comparable across clients.
The threshold of that node is ``split_values[e, i, argmax_v]`` (in the preprocessed scale: numeric features are
z-scored, so thresholds on numeric features are in training-fold SD units; one-hot features are 0/1).
"""
from __future__ import annotations

import numpy as np
import torch


def selections(split_index_array, split_values, features_by_estimator):
    sia = torch.as_tensor(split_index_array)
    arg = sia.argmax(-1)                                                    # [E, I]
    fbe = torch.as_tensor(features_by_estimator)
    feat = torch.gather(fbe, 1, arg)                                        # [E, I] (fbe is [E, V], arg indexes V)
    thr = torch.gather(torch.as_tensor(split_values), 2, arg.unsqueeze(-1)).squeeze(-1)
    return feat.numpy(), thr.numpy()


def consensus(client_feats: np.ndarray, client_thr: np.ndarray, global_feat: np.ndarray | None, n_numeric: int) -> dict:
    """client_feats/client_thr: [K, E, I]; global_feat: [E, I] or None."""
    K = client_feats.shape[0]
    F = client_feats.reshape(K, -1)                                         # [K, N]
    T = client_thr.reshape(K, -1)
    N = F.shape[1]
    modal = np.empty(N, dtype=np.int64)
    frac = np.empty(N)
    thr_sd = np.full(N, np.nan)
    for j in range(N):
        vals, counts = np.unique(F[:, j], return_counts=True)
        k = int(np.argmax(counts))                                          # ties -> smallest feature id
        modal[j], frac[j] = vals[k], counts[k] / K
        same = F[:, j] == vals[k]
        if same.sum() >= 2:
            thr_sd[j] = float(np.std(T[same, j]))
    is_num = modal < n_numeric
    out = {
        "n_clients": int(K), "n_nodes": int(N),
        "consensus_mean": float(frac.mean()), "consensus_median": float(np.median(frac)),
        "frac_nodes_unanimous": float((frac == 1.0).mean()),
        "frac_nodes_modal_numeric": float(is_num.mean()),
        "thr_sd_numeric_nodes_mean": float(np.nanmean(thr_sd[is_num])) if np.isfinite(thr_sd[is_num]).any() else float("nan"),
        "thr_sd_numeric_nodes_median": float(np.nanmedian(thr_sd[is_num])) if np.isfinite(thr_sd[is_num]).any() else float("nan"),
        "thr_sd_onehot_nodes_mean": float(np.nanmean(thr_sd[~is_num])) if np.isfinite(thr_sd[~is_num]).any() else float("nan"),
    }
    if global_feat is not None:
        out["global_agreement"] = float((F == global_feat.reshape(1, -1)).mean())
    return out


def update_disagreement(client_states: list, before: dict, weights: np.ndarray, keys=None) -> dict:
    keys = keys or list(before)
    D = np.stack([torch.cat([(s[k] - before[k]).flatten() for k in keys]).numpy() for s in client_states])   # [K, P]
    w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    mean_upd = (w[:, None] * D).sum(0)
    norms = np.linalg.norm(D, axis=1)
    mnorm = np.linalg.norm(mean_upd)
    cos = (D @ mean_upd) / np.maximum(norms * mnorm, 1e-12)
    return {"update_norm_mean": float(norms.mean()), "update_norm_sd": float(norms.std()),
            "aggregate_update_norm": float(mnorm), "cos_to_aggregate_mean": float(cos.mean()),
            "cos_to_aggregate_min": float(cos.min()),
            "update_norm_ratio_aggregate_over_mean": float(mnorm / max(norms.mean(), 1e-12))}
