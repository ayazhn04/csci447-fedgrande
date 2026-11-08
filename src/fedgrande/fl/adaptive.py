"""Phase 2F: Consensus-Gated progressive Structural Freezing (CGSF) for FedAvg-GRANDE -- our experimental method.

Same FedAvg-with-local-Adam loop as ``train.fedavg`` (rounds without frozen slices are numerically identical), plus
per-node structural masks over the [estimator, internal_node, :] slices of ``split_index_array`` (feature logits) and
``split_values`` (thresholds). Leaves and estimator weights are always trained and aggregated.

mode "global" (G1): after round `warmup`, if mean node consensus >= c_mean AND unanimous fraction >= c_unan for
    `consecutive` rounds, all nodes become fully frozen (both tensors) from the next round on.
mode "nodewise" (G2): Stage A, node j feature-frozen after C_j >= c_node for `consecutive` rounds (rounds > warmup);
    Stage B, a feature-frozen node becomes fully frozen after D_j <= d_max and S_j <= s_max for `consecutive` rounds
    counted only in rounds that start with the node already feature-frozen.
Decisions use only clients' locally trained structural parameters, global parameters and training-row feature SDs.
A decision taken after round r applies from round r+1; the newly frozen slices are broadcast once more in round r+1's
downlink (clients do not yet hold the aggregated round-r value) and never again; uplink excludes them from round r+1.
Frozen slices: gradients zeroed before every optimizer step (fresh Adam each round => moments stay exactly 0) and
asserted bitwise unchanged after every step; aggregation copies them from the previous global state.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import diagnostics as diag
from .metrics import safe_auc
from .models import FACTORY, GRANDE_BETAS, GRANDE_LR, buffer_state, load_trainable, trainable_state
from .train import Stopper, _t, gen_for, predict, weighted_average, weighted_loss

STRUCT = ("split_index_array", "split_values")
OUTPUT = ("leaf_classes_array", "estimator_weights")


def node_consensus(F: np.ndarray) -> np.ndarray:
    """F: [K, E, I] selected feature ids per client -> [E, I] fraction of clients choosing the modal feature."""
    K = F.shape[0]
    same = (F[:, None] == F[None, :]).sum(1)          # [K, E, I]: for each client, how many clients agree with it
    return same.max(0) / K


def masked_epoch(model, opt, X, y, w, gen, batch_size, feat_mask, full_mask, frozen_ref):
    """One local epoch; frozen slices get zero gradients and are asserted unchanged after every step."""
    import torch.nn.functional as F
    n = len(X)
    bs = max(1, min(batch_size, n // 2))
    perm = torch.randperm(n, generator=gen)
    model.train()
    any_feat, any_full = bool(feat_mask.any()), bool(full_mask.any())
    tot, cnt = 0.0, 0
    for s in range(0, n - bs + 1, bs):
        idx = perm[s:s + bs]
        loss = F.cross_entropy(model(X[idx]), y[idx], weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if any_feat and model.split_index_array.grad is not None:
            model.split_index_array.grad[feat_mask] = 0.0
        if any_full and model.split_values.grad is not None:
            model.split_values.grad[full_mask] = 0.0
        opt.step()
        if any_feat:
            assert torch.equal(model.split_index_array.detach()[feat_mask], frozen_ref["split_index_array"][feat_mask]), "frozen feature slice moved"
        if any_full:
            assert torch.equal(model.split_values.detach()[full_mask], frozen_ref["split_values"][full_mask]), "frozen threshold slice moved"
        tot += float(loss.item()) * len(idx)
        cnt += len(idx)
    return tot / max(cnt, 1), cnt


def fedavg_adaptive(client_data, Xval_pooled, yval_pooled, seed, cfg, w, n_numeric, feature_sd):
    ad = cfg["adaptive"]
    build, make_opt = FACTORY["grande"]
    D = client_data[0]["Xfit"].shape[1]
    glob = build(D, seed, **cfg.get("model_kwargs", {}))
    local = build(D, seed, **cfg.get("model_kwargs", {}))
    local.load_state_dict(glob.state_dict())
    init_buffers = buffer_state(glob)
    fbe = glob.features_by_estimator
    E, I, V = glob.split_values.shape
    n_fit = np.array([len(d["Xfit"]) for d in client_data], dtype=np.float64)
    active = n_fit >= 2
    weights = np.where(active, n_fit, 0.0)
    n_act = int(active.sum())
    data = [(_t(d["Xfit"]), _t(d["yfit"], torch.long)) for d in client_data]
    Xv = _t(Xval_pooled)
    sd = torch.as_tensor(np.asarray(feature_sd, dtype=np.float64))
    eps = ad.get("eps", 1e-6)

    feat_frozen = torch.zeros(E, I, dtype=torch.bool)
    full_frozen = torch.zeros(E, I, dtype=torch.bool)
    newly_feat = torch.zeros(E, I, dtype=torch.bool)          # frozen at end of previous round -> broadcast once more
    newly_full = torch.zeros(E, I, dtype=torch.bool)
    cnt_feat = torch.zeros(E, I, dtype=torch.int64)
    cnt_full = torch.zeros(E, I, dtype=torch.int64)
    feat_round = torch.full((E, I), -1, dtype=torch.int64)
    full_round = torch.full((E, I), -1, dtype=torch.int64)
    g_cnt, trigger_round = 0, None
    prev_thr = None
    st = Stopper(cfg["patience"], cfg["warmup"])
    rounds = []
    uplink_int8 = ad.get("uplink") == "int8"

    for r in range(1, cfg["max_rounds"] + 1):
        t0 = time.perf_counter()
        before = trainable_state(glob)
        feat_at_start, full_at_start = feat_frozen.clone(), full_frozen.clone()
        whole_sia, whole_sv = bool(feat_frozen.all()), bool(full_frozen.all())
        client_states, losses = [], []
        for c, (X, y) in enumerate(data):
            if not active[c]:
                client_states.append(before)
                losses.append(np.nan)
                continue
            local.load_state_dict(glob.state_dict())
            if whole_sia or whole_sv:                          # whole tensor frozen: drop it from autograd/optimizer
                for n_, p_ in local.named_parameters():
                    p_.requires_grad_(not ((n_ == "split_index_array" and whole_sia) or (n_ == "split_values" and whole_sv)))
                opt = torch.optim.Adam([{"params": getattr(local, n), "lr": lr, "betas": GRANDE_BETAS} for n, lr in GRANDE_LR.items()
                                        if getattr(local, n).requires_grad])
            else:
                opt = make_opt(local)
            ref = {k: before[k] for k in STRUCT}
            tl, _ = masked_epoch(local, opt, X, y, w, gen_for(seed, r, c), cfg["batch_size"],
                                 feat_mask=feat_frozen if not whole_sia else torch.zeros_like(feat_frozen),
                                 full_mask=full_frozen if not whole_sv else torch.zeros_like(full_frozen), frozen_ref=ref)
            for p_ in local.parameters():
                p_.requires_grad_(True)
            losses.append(tl)
            cs = {n_: p_.detach().clone() for n_, p_ in local.named_parameters()}
            assert torch.equal(cs["split_index_array"][feat_frozen], before["split_index_array"][feat_frozen])
            assert torch.equal(cs["split_values"][full_frozen], before["split_values"][full_frozen])
            client_states.append(cs)

        sent = client_states
        qerr = None
        if uplink_int8:
            from . import compress
            sent, qerr = [], []
            for c in range(len(data)):
                if not active[c]:
                    sent.append(client_states[c])
                    continue
                hat, rel = compress.roundtrip_update(client_states[c], before)
                sent.append(hat)
                qerr.append(rel)
        new = weighted_average(sent, weights)
        new["split_index_array"][feat_frozen] = before["split_index_array"][feat_frozen]   # clients cannot overwrite frozen slices
        new["split_values"][full_frozen] = before["split_values"][full_frozen]
        load_trainable(glob, new)
        after = trainable_state(glob)
        assert torch.equal(after["split_index_array"][feat_frozen], before["split_index_array"][feat_frozen])
        assert torch.equal(after["split_values"][full_frozen], before["split_values"][full_frozen])

        # ---- communication of this round (slice level); newly frozen slices from last round's decision still go down once
        n_sia_up = int((~feat_at_start).sum()) * V
        n_sv_up = int((~full_at_start).sum()) * V
        n_sia_down = n_sia_up + int(newly_feat.sum()) * V
        n_sv_down = n_sv_up + int(newly_full.sum()) * V
        n_out = sum(before[k].numel() for k in OUTPUT)
        n_leaf, n_ew = before["leaf_classes_array"].numel(), before["estimator_weights"].numel()
        rec = {"round": r, "bytes_down_sia": 4 * n_sia_down * n_act, "bytes_down_sv": 4 * n_sv_down * n_act,
               "bytes_down_leaf": 4 * n_leaf * n_act, "bytes_down_ew": 4 * n_ew * n_act}
        if uplink_int8:   # active elements as int8 + one fp32 scale per transmitted tensor
            rec.update({"bytes_up_sia": (n_sia_up + (4 if n_sia_up else 0)) * n_act, "bytes_up_sv": (n_sv_up + (4 if n_sv_up else 0)) * n_act,
                        "bytes_up_leaf": (n_leaf + 4) * n_act, "bytes_up_ew": (n_ew + 4) * n_act})
        else:
            rec.update({"bytes_up_sia": 4 * n_sia_up * n_act, "bytes_up_sv": 4 * n_sv_up * n_act,
                        "bytes_up_leaf": 4 * n_leaf * n_act, "bytes_up_ew": 4 * n_ew * n_act})
        rec["bytes_down"] = rec["bytes_down_sia"] + rec["bytes_down_sv"] + rec["bytes_down_leaf"] + rec["bytes_down_ew"]
        rec["bytes_up"] = rec["bytes_up_sia"] + rec["bytes_up_sv"] + rec["bytes_up_leaf"] + rec["bytes_up_ew"]
        assert n_out == n_leaf + n_ew

        # ---- diagnostics + freeze decisions (training-side information only)
        act = [c for c in range(len(data)) if active[c]]
        sels = [diag.selections(client_states[c]["split_index_array"], client_states[c]["split_values"], fbe) for c in act]
        Fk = np.stack([s[0] for s in sels])                    # [K, E, I]
        Tk = np.stack([s[1] for s in sels])
        Cj = torch.as_tensor(node_consensus(Fk))
        gfeat, gthr = diag.selections(after["split_index_array"], after["split_values"], fbe)
        rec.update({"cons_mean": float(Cj.mean()), "cons_unanimous": float((Cj == 1.0).float().mean()),
                    "frac_feat_frozen": float(feat_at_start.float().mean()), "frac_full_frozen": float(full_at_start.float().mean())})
        newly_feat = torch.zeros(E, I, dtype=torch.bool)
        newly_full = torch.zeros(E, I, dtype=torch.bool)
        if r > ad["warmup"]:
            if ad["mode"] == "global":
                ok = rec["cons_mean"] >= ad["c_mean"] and rec["cons_unanimous"] >= ad["c_unan"]
                g_cnt = g_cnt + 1 if ok else 0
                if trigger_round is None and g_cnt >= ad["consecutive"]:
                    trigger_round = r
                    newly_feat, newly_full = ~feat_frozen, ~full_frozen
                    feat_round[~feat_frozen] = r
                    full_round[~full_frozen] = r
                    feat_frozen[:] = True
                    full_frozen[:] = True
            else:
                okf = (Cj >= ad["c_node"]) & ~feat_frozen
                cnt_feat = torch.where(okf, cnt_feat + 1, torch.zeros_like(cnt_feat))
                # threshold gate only for nodes that started this round feature-frozen
                gt = torch.as_tensor(gthr, dtype=torch.float64)
                scale = sd[torch.as_tensor(gfeat)] + eps
                Dj = (gt - prev_thr).abs() / scale if prev_thr is not None else torch.full_like(gt, float("inf"))
                Sj = torch.as_tensor(Tk.std(0), dtype=torch.float64) / scale
                okt = feat_at_start & ~full_frozen & (Dj <= ad["d_max"]) & (Sj <= ad["s_max"])
                cnt_full = torch.where(okt, cnt_full + 1, torch.zeros_like(cnt_full))
                newly_full = (cnt_full >= ad["consecutive"]) & ~full_frozen
                newly_feat = (cnt_feat >= ad["consecutive"]) & ~feat_frozen
                feat_round[newly_feat] = r
                full_round[newly_full] = r
                feat_frozen |= newly_feat
                full_frozen |= newly_full
                assert not (full_frozen & ~feat_frozen).any()
                rec["thr_D_median_featfrozen"] = float(Dj[feat_at_start].median()) if feat_at_start.any() else float("nan")
                rec["thr_S_median_featfrozen"] = float(Sj[feat_at_start].median()) if feat_at_start.any() else float("nan")
        prev_thr = torch.as_tensor(gthr, dtype=torch.float64)
        rec["newly_feat_frozen"], rec["newly_full_frozen"] = int(newly_feat.sum()), int(newly_full.sum())
        for k, b in buffer_state(glob).items():
            assert torch.equal(b, init_buffers[k]), f"buffer {k} changed"
        if qerr:
            rec["qerr_rel_l2_max_all"] = float(max(v for e in qerr for v in e.values()))
        rec["train_loss_weighted"] = float(np.nansum(np.array(losses) * (weights / weights.sum())))
        rec["val_auc"] = safe_auc(yval_pooled, predict(glob, Xv))
        rec["val_loss"] = weighted_loss(glob, Xv, _t(yval_pooled, torch.long), w)
        rec["seconds_total"] = time.perf_counter() - t0
        rounds.append(rec)
        if st.update(r, rec["val_auc"], glob):
            break
    glob.load_state_dict(st.state)
    return glob, {"best_step": st.best_step, "steps_executed": r, "best_val_auc": st.best, "rounds": rounds,
                  "client_weights": weights.tolist(), "inactive_clients": [int(c) for c in np.flatnonzero(~active)],
                  "init_buffers": init_buffers, "trigger_round": trigger_round,
                  "feat_freeze_round": feat_round.numpy(), "full_freeze_round": full_round.numpy()}
