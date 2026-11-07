"""Phase 2C step 0: deterministically regenerate the 15 Phase 2B FedAvg-GRANDE Dirichlet(0.1) global models
(Phase 2B saved predictions/rounds/selections but not model states) and verify them BITWISE against Phase 2B.

Uses the unchanged Phase 2B job code (run_job.py, only an optional --save-state flag added). A checkpoint is accepted only
if the regenerated validation/test probabilities, the per-round validation AUC/loss and the selected split features and
thresholds are all bitwise identical to the Phase 2B artifacts.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.provenance import sha256_file, write_json  # noqa: E402

P2B = ROOT / "results" / "phase2b" / "runs"
REGEN = ROOT / "results" / "phase2c" / "regenerated_global"
CKPT = ROOT / "data" / "derived" / "phase2c" / "global_ckpt"


def main():
    rows = []
    for seed in (42, 447, 2026):
        for fold in range(5):
            jid = f"fedavg_grande__dirichlet0.1__f{fold}__s{seed}"
            ck = CKPT / f"{jid}.pt"
            if not ck.exists():
                subprocess.run([sys.executable, "-I", str(ROOT / "src/fedgrande/phase2b/run_job.py"), "--method", "fedavg_grande",
                                "--partition", "dirichlet0.1", "--fold", str(fold), "--seed", str(seed), "--out-root", str(REGEN),
                                "--save-state", str(ck)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            a, b = pd.read_csv(P2B / jid / "predictions.csv"), pd.read_csv(REGEN / jid / "predictions.csv")
            ra, rb = pd.read_csv(P2B / jid / "rounds.csv"), pd.read_csv(REGEN / jid / "rounds.csv")
            sa, sb = np.load(P2B / jid / "selections.npz"), np.load(REGEN / jid / "selections.npz")
            ja, jb = json.load(open(P2B / jid / "result.json")), json.load(open(REGEN / jid / "result.json"))
            rec = {"job": jid, "fold": fold, "seed": seed,
                   "predictions_bitwise_equal": bool(a[["row_index", "role", "y", "client"]].equals(b[["row_index", "role", "y", "client"]])
                                                     and np.array_equal(a.p1.to_numpy(), b.p1.to_numpy())),
                   "rounds_val_auc_equal": bool(np.array_equal(ra.val_auc.to_numpy(), rb.val_auc.to_numpy())),
                   "rounds_val_loss_equal": bool(np.array_equal(ra.val_loss.to_numpy(), rb.val_loss.to_numpy())),
                   "selections_equal": bool(np.array_equal(sa["feat"], sb["feat"]) and np.array_equal(sa["thr"], sb["thr"])),
                   "best_round_equal": ja["best_step"] == jb["best_step"], "best_round": jb["best_step"],
                   "checkpoint": str(ck.relative_to(ROOT)), "checkpoint_sha256": sha256_file(ck)}
            rec["verified"] = all(v for k, v in rec.items() if k.endswith("equal"))
            rows.append(rec)
            print(rec["job"], rec["verified"], flush=True)
    write_json(ROOT / "results" / "phase2c" / "checkpoint_regeneration.json",
               {"all_verified": all(r["verified"] for r in rows), "checkpoints": rows,
                "note": "Phase 2B did not persist model states; checkpoints regenerated with identical code and verified bitwise against Phase 2B artifacts"})
    print("ALL VERIFIED:", all(r["verified"] for r in rows))


if __name__ == "__main__":
    main()
