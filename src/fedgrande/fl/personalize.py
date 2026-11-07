"""Phase 2C: client-side adaptation of a frozen global FedAvg-GRANDE model (no extra communication).

P1  client intercept calibration: logit(p_c) = logit(p_global) + b_c, b_c = argmin_b  sum_i CE(y_i, sigmoid(z_i + b)) + (lam/2) b^2
    over the client's FIT rows only (unweighted CE; L2/Gaussian-prior regularization keeps b finite for single-class
    clients). Solved by Newton's method (deterministic). Cannot change within-client ranking.
P2  personalized leaves: only ``leaf_classes_array`` is trainable.
P3  personalized leaves + estimator weights: ``leaf_classes_array`` and ``estimator_weights`` trainable.
    In P2/P3 ``split_index_array``, ``split_values`` and all buffers are frozen (requires_grad=False) and verified bitwise.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import safe_auc
from .train import gen_for

# GRANDE binary output is logits [-r, r]; softmax -> p1 = sigmoid(2r). "logit(p)" below means log(p/(1-p)) = 2r.
TRAINABLE = {"P2": ("leaf_classes_array",), "P3": ("leaf_classes_array", "estimator_weights")}
FROZEN_ALWAYS = ("split_index_array", "split_values")


def logit_from_p(p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def fit_intercept(z: np.ndarray, y: np.ndarray, lam: float, iters: int = 100, tol: float = 1e-12) -> float:
    """Penalized 1-D logistic intercept on fixed offsets z (Newton). Always finite for lam > 0."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(z + b)))
        g = np.sum(p - y) + lam * b
        h = np.sum(p * (1 - p)) + lam
        step = g / h
        b -= step
        if abs(step) < tol:
            break
    return float(b)


def apply_intercept(p_global, b):
    return 1.0 / (1.0 + np.exp(-(logit_from_p(p_global) + b)))


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    if len(X) == 0:
        return np.zeros(0)
    return np.concatenate([F.softmax(model(X[i:i + bs]), -1)[:, 1].numpy() for i in range(0, len(X), bs)]).astype(np.float64)


@torch.no_grad()
def ce_loss(model, X, y) -> float:
    model.eval()
    return float(F.cross_entropy(model(X), y).item())


def personalize(global_model, variant: str, Xfit, yfit, Xval, yval, seed: int, client: int, cfg: dict):
    """Fine-tune only the variant's tensors on the client's fit rows (unweighted CE, Adam), select on client validation.

    Selection: client validation ROC-AUC if the client's validation rows contain both classes, otherwise validation
    cross-entropy (deterministic fallback). Epoch 0 (= unpersonalized global state) is a candidate, so the restored state
    is never worse than the global model on the selection criterion. Returns (model, info).
    """
    m = copy.deepcopy(global_model)
    names = TRAINABLE[variant]
    for n, p in m.named_parameters():
        p.requires_grad_(n in names)
    opt = torch.optim.Adam([{"params": getattr(m, n), "lr": cfg["lr"][n], "betas": tuple(cfg["betas"])} for n in names])
    X, y = torch.as_tensor(Xfit), torch.as_tensor(yfit, dtype=torch.long)
    Xv, yv = torch.as_tensor(Xval), torch.as_tensor(yval, dtype=torch.long)
    rule = "val_auc" if len(np.unique(yval)) == 2 else "val_loss"

    def score():
        return safe_auc(yval, predict(m, Xv)) if rule == "val_auc" else -ce_loss(m, Xv, yv)  # higher is better

    best, best_ep, best_state = score(), 0, copy.deepcopy(m.state_dict())
    n = len(X)
    bs = max(1, min(cfg["batch_size"], n // 2))
    ep = 0
    for ep in range(1, cfg["max_epochs"] + 1):
        m.train()
        perm = torch.randperm(n, generator=gen_for(seed, 7_000 + client, ep))
        for s in range(0, n - bs + 1, bs):
            idx = perm[s:s + bs]
            loss = F.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sc = score()
        if sc > best + cfg["min_delta"]:
            best, best_ep, best_state = sc, ep, copy.deepcopy(m.state_dict())
        if ep - best_ep >= cfg["patience"]:
            break
    m.load_state_dict(best_state)
    for p in m.parameters():
        p.requires_grad_(True)
    return m, {"rule": rule, "best_epoch": best_ep, "epochs_run": ep, "best_score": float(best), "n_fit": int(n), "n_val": int(len(Xv))}


def verify_structure(global_model, model, variant: str) -> dict:
    """Bitwise checks: frozen tensors and all buffers unchanged; reports which parameters changed."""
    g, s = global_model.state_dict(), model.state_dict()
    frozen = list(FROZEN_ALWAYS) + [n for n, _ in global_model.named_buffers()]
    if variant == "P2":
        frozen.append("estimator_weights")
    out = {f"unchanged__{k}": bool(torch.equal(g[k], s[k])) for k in frozen}
    out["changed_params"] = [k for k, _ in global_model.named_parameters() if not torch.equal(g[k], s[k])]
    out["only_intended_changed"] = set(out["changed_params"]) <= set(TRAINABLE[variant]) and all(v for k, v in out.items() if k.startswith("unchanged__"))
    return out
