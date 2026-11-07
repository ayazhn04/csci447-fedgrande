"""Training procedures shared by GRANDE and MLP: centralized, local-only and FedAvg with local Adam.

Common rules (frozen in configs/phase2b/core.json):
* loss: class-weighted cross-entropy (weights from the GLOBAL training-fold class counts, GRANDE formula);
* batch size per data holder: max(1, min(256, n // 2)), shuffled, drop_last (GRANDE 1.0.0 rule);
* model selection: pooled VALIDATION ROC-AUC after every epoch (centralized/local) or round (FedAvg);
  improvement = AUC > best + 1e-4; stop when epoch >= warmup and no improvement for `patience`; restore best state;
* FedAvg: full participation, 1 local epoch per round, fresh local Adam each round ("FedAvg with local Adam"),
  aggregation weighted by client training-row count, trainable parameters only (buffers stay identical).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import diagnostics as diag
from .metrics import safe_auc
from .models import FACTORY, buffer_state, load_trainable, trainable_state


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_train.astype(int), minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    return torch.tensor(inv / inv.sum(), dtype=torch.float32)


def weighted_average(states: list, weights) -> dict:
    """FedAvg aggregation: sum_c (w_c / sum w) * theta_c for every trainable tensor (clients with weight 0 ignored)."""
    w = np.asarray(weights, dtype=np.float64)
    wn = w / w.sum()
    keys = states[0].keys()
    return {k: sum(float(wn[c]) * states[c][k] for c in range(len(states)) if wn[c] > 0) for k in keys}


def gen_for(seed: int, *ids) -> torch.Generator:
    v = int(seed)
    for i in ids:
        v = (v * 1_000_003 + int(i) + 1) % (2 ** 62)
    return torch.Generator().manual_seed(v)


def train_one_epoch(model, opt, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, gen: torch.Generator, batch_size=256):
    n = len(X)
    bs = max(1, min(batch_size, n // 2))
    perm = torch.randperm(n, generator=gen)
    model.train()
    tot, cnt = 0.0, 0
    for s in range(0, n - bs + 1, bs):
        idx = perm[s:s + bs]
        loss = F.cross_entropy(model(X[idx]), y[idx], weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += float(loss.item()) * len(idx)
        cnt += len(idx)
    return tot / max(cnt, 1), cnt


@torch.no_grad()
def predict(model, X: torch.Tensor, bs=512) -> np.ndarray:
    model.eval()
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([F.softmax(model(X[i:i + bs]), -1)[:, 1].numpy() for i in range(0, len(X), bs)])


@torch.no_grad()
def weighted_loss(model, X, y, w) -> float:
    model.eval()
    return float(F.cross_entropy(model(X), y, weight=w).item()) if len(X) else float("nan")


class Stopper:
    def __init__(self, patience, warmup, mode="max", min_delta=1e-4):
        self.patience, self.warmup, self.mode, self.min_delta = patience, warmup, mode, min_delta
        self.best, self.best_step, self.state = None, None, None

    def update(self, step, value, model) -> bool:
        better = self.best is None or (value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta)
        if better and np.isfinite(value):
            self.best, self.best_step, self.state = value, step, copy.deepcopy(model.state_dict())
        return step >= self.warmup and self.best_step is not None and step - self.best_step >= self.patience


def _t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def centralized(kind, Xfit, yfit, Xval, yval, seed, cfg, w):
    build, make_opt = FACTORY[kind]
    model = build(Xfit.shape[1], seed, **cfg.get("model_kwargs", {}))
    opt = make_opt(model)
    X, y, Xv, yv = _t(Xfit), _t(yfit, torch.long), _t(Xval), _t(yval, torch.long)
    st = Stopper(cfg["patience"], cfg["warmup"])
    curve = []
    for ep in range(1, cfg["max_epochs"] + 1):
        t0 = time.perf_counter()
        tl, _ = train_one_epoch(model, opt, X, y, w, gen_for(seed, ep), cfg["batch_size"])
        auc = safe_auc(yval, predict(model, Xv))
        curve.append({"step": ep, "train_loss": tl, "val_auc": auc, "val_loss": weighted_loss(model, Xv, yv, w),
                      "seconds": time.perf_counter() - t0})
        if st.update(ep, auc, model):
            break
    model.load_state_dict(st.state)
    return model, {"best_step": st.best_step, "steps_executed": ep, "best_val_auc": st.best, "curve": curve, "stop_rule": "val_auc"}


def local_only(kind, client_data, seed, cfg, w):
    """client_data: list of dicts with Xfit,yfit,Xval,yval (numpy). Returns models and per-client info."""
    build, make_opt = FACTORY[kind]
    models, infos = [None] * len(client_data), [None] * len(client_data)
    deferred = []
    for c, d in enumerate(client_data):
        model = build(d["Xfit"].shape[1], seed, **cfg.get("model_kwargs", {}))
        X, y = _t(d["Xfit"]), _t(d["yfit"], torch.long)
        Xv, yv = _t(d["Xval"]), _t(d["yval"], torch.long)
        n_fit, n_val, val_classes = len(X), len(Xv), len(np.unique(d["yval"]))
        if n_fit < 2:
            models[c], infos[c] = model, {"client": c, "rule": "untrained_lt2_fit_rows", "best_step": 0, "steps_executed": 0, "n_fit": n_fit, "n_val": n_val}
            continue
        if n_val == 0:
            deferred.append(c)
            continue
        rule = "val_auc" if val_classes == 2 else "val_loss"
        opt = make_opt(model)
        st = Stopper(cfg["patience"], cfg["warmup"], mode="max" if rule == "val_auc" else "min")
        for ep in range(1, cfg["max_epochs"] + 1):
            train_one_epoch(model, opt, X, y, w, gen_for(seed, c, ep), cfg["batch_size"])
            val = safe_auc(d["yval"], predict(model, Xv)) if rule == "val_auc" else weighted_loss(model, Xv, yv, w)
            if st.update(ep, val, model):
                break
        model.load_state_dict(st.state)
        models[c] = model
        infos[c] = {"client": c, "rule": rule, "best_step": st.best_step, "steps_executed": ep, "n_fit": n_fit, "n_val": n_val,
                    "fit_pos": int(d["yfit"].sum()), "val_pos": int(d["yval"].sum())}
    if deferred:
        done = [i["best_step"] for i in infos if i is not None and i["rule"] in ("val_auc", "val_loss")]
        k = int(np.median(done)) if done else cfg["warmup"]
        for c in deferred:
            d = client_data[c]
            model = build(d["Xfit"].shape[1], seed, **cfg.get("model_kwargs", {}))
            opt = make_opt(model)
            X, y = _t(d["Xfit"]), _t(d["yfit"], torch.long)
            for ep in range(1, k + 1):
                train_one_epoch(model, opt, X, y, w, gen_for(seed, c, ep), cfg["batch_size"])
            models[c] = model
            infos[c] = {"client": c, "rule": "no_val_rows_median_best_epoch_of_other_clients", "best_step": k, "steps_executed": k,
                        "n_fit": len(X), "n_val": 0, "fit_pos": int(d["yfit"].sum()), "val_pos": 0}
    return models, infos


def fedavg(kind, client_data, Xval_pooled, yval_pooled, seed, cfg, w, n_numeric=None, on_round=None):
    """FedAvg with local Adam. Returns (global model at best round, info with per-round records)."""
    build, make_opt = FACTORY[kind]
    D = client_data[0]["Xfit"].shape[1]
    glob = build(D, seed, **cfg.get("model_kwargs", {}))
    local = build(D, seed, **cfg.get("model_kwargs", {}))
    local.load_state_dict(glob.state_dict())
    init_buffers = buffer_state(glob)
    n_fit = np.array([len(d["Xfit"]) for d in client_data], dtype=np.float64)
    active = n_fit >= 2
    weights = np.where(active, n_fit, 0.0)
    data = [(_t(d["Xfit"]), _t(d["yfit"], torch.long)) for d in client_data]
    Xv = _t(Xval_pooled)
    st = Stopper(cfg["patience"], cfg["warmup"])
    rounds = []
    fz = cfg.get("freeze")                                    # Phase 2E structure-freezing schedule (None = ordinary FedAvg)
    snapshots = {}
    for r in range(1, cfg["max_rounds"] + 1):
        t0 = time.perf_counter()
        before = trainable_state(glob)
        frozen = list(fz["params"]) if (fz and r > fz["after_round"]) else []
        act_keys = [k for k in before if k not in frozen]
        client_states, losses = [], []
        for c, (X, y) in enumerate(data):
            if not active[c]:
                client_states.append(before)
                losses.append(np.nan)
                continue
            local.load_state_dict(glob.state_dict())          # identical global state incl. buffers
            if fz:
                for n_, p_ in local.named_parameters():
                    p_.requires_grad_(n_ not in frozen)        # frozen tensors: no gradients
                from .models import grande_optimizer_subset
                opt = grande_optimizer_subset(local, act_keys) if frozen else make_opt(local)
            else:
                opt = make_opt(local)                          # fresh optimizer state every round
            for _ in range(cfg["local_epochs"]):
                tl, _ = train_one_epoch(local, opt, X, y, w, gen_for(seed, r, c), cfg["batch_size"])
            losses.append(tl)
            client_states.append({n_: p_.detach().clone() for n_, p_ in local.named_parameters()} if fz else trainable_state(local))
            for k in frozen:
                assert torch.equal(client_states[-1][k], before[k]), f"frozen tensor {k} changed on client {c}"
        sent_states = client_states
        uplink_now = (fz.get("uplink_after", "fp32") if frozen else "fp32") if fz else cfg.get("uplink", "fp32")
        if uplink_now == "int8":                               # Phase 2D codec: quantized client updates (uplink only)
            from . import compress
            sent_states, qerr = [], []
            for c in range(len(data)):
                if not active[c]:
                    sent_states.append(client_states[c])
                    continue
                sub_hat, rel = compress.roundtrip_update({k: client_states[c][k] for k in act_keys}, {k: before[k] for k in act_keys})
                sent_states.append({**client_states[c], **sub_hat})
                qerr.append(rel)
        wn = weights / weights.sum()
        if frozen:                                             # aggregate active tensors only; frozen copied unchanged
            agg = weighted_average([{k: st_[k] for k in act_keys} for st_ in sent_states], weights)
            load_trainable(glob, {**agg, **{k: before[k] for k in frozen}})
        else:
            load_trainable(glob, weighted_average(sent_states, weights))
        t_train = time.perf_counter() - t0
        rec = {"round": r, "train_loss_weighted": float(np.nansum(np.array(losses) * wn)), "seconds_train": t_train}
        if fz:
            from . import compress
            from .comm import tensor_bytes
            after = trainable_state(glob)
            n_act = int(active.sum())
            down_keys = list(before) if (not frozen or r == fz["after_round"] + 1) else act_keys   # round after_round+1 re-broadcasts structure once
            rec["active_groups"] = "|".join(act_keys)
            rec["bytes_down"] = n_act * tensor_bytes({k: before[k] for k in down_keys})
            rec["bytes_up"] = n_act * (compress.payload_bytes({k: before[k] for k in act_keys}) if uplink_now == "int8"
                                       else tensor_bytes({k: before[k] for k in act_keys}))
            for k in before:
                rec[f"move_rel__{k}"] = float(torch.linalg.vector_norm(after[k] - before[k]) / torch.linalg.vector_norm(before[k]))
            for k in frozen:
                assert torch.equal(after[k], before[k]), f"global frozen tensor {k} changed"
        if r in (cfg.get("snapshot_rounds") or ()):
            snapshots[r] = copy.deepcopy(glob.state_dict())
        if uplink_now == "int8":
            for k in act_keys:
                vals = [e[k] for e in qerr]
                rec[f"qerr_rel_l2_mean__{k}"], rec[f"qerr_rel_l2_max__{k}"] = float(np.mean(vals)), float(np.max(vals))
            allv = [v for e in qerr for v in e.values()]
            rec["qerr_rel_l2_mean_all"], rec["qerr_rel_l2_max_all"] = float(np.mean(allv)), float(np.max(allv))
        rec["val_auc"] = safe_auc(yval_pooled, predict(glob, Xv))
        rec["val_loss"] = weighted_loss(glob, Xv, _t(yval_pooled, torch.long), w)
        act = [c for c in range(len(data)) if active[c]]
        rec.update(diag.update_disagreement([client_states[c] for c in act], before, weights[act]))
        if kind == "grande":
            fbe = glob.features_by_estimator
            sel = [diag.selections(client_states[c]["split_index_array"], client_states[c]["split_values"], fbe) for c in act]
            gfeat, _ = diag.selections(glob.split_index_array.detach(), glob.split_values.detach(), fbe)
            cons = diag.consensus(np.stack([s[0] for s in sel]), np.stack([s[1] for s in sel]), gfeat, n_numeric)
            rec.update({f"cons_{k}": v for k, v in cons.items()})
            rec["update_disagreement_split_index"] = diag.update_disagreement([client_states[c] for c in act], before, weights[act],
                                                                               keys=["split_index_array"])["cos_to_aggregate_mean"]
        for k, b in buffer_state(glob).items():
            assert torch.equal(b, init_buffers[k]), f"buffer {k} changed"
        rec["seconds_total"] = time.perf_counter() - t0
        rounds.append(rec)
        if on_round:
            on_round(rec)
        if st.update(r, rec["val_auc"], glob):
            break
    glob.load_state_dict(st.state)
    return glob, {"best_step": st.best_step, "steps_executed": r, "best_val_auc": st.best, "rounds": rounds, "snapshots": snapshots,
                  "client_weights": weights.tolist(), "inactive_clients": [int(c) for c in np.flatnonzero(~active)],
                  "init_buffers": init_buffers}
