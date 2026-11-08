"""Run the frozen Phase 2B matrix serially, one subprocess per job (resumable: skips jobs with result.json).

Usage: python -I src/fedgrande/phase2b/run_matrix.py --stage core [--dry-run]
Stage 'core': methods x partitions {natural, iid} x C2 folds 0-4 x seeds {42, 447, 2026}.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.phase2b.run_job import job_id  # noqa: E402


def jobs(cfg, stage):
    out = []
    parts = {"core": ["natural", "iid"], "dirichlet": ["dirichlet0.1"], "compression": ["natural", "iid"], "structure": ["natural", "iid"], "adaptive": ["natural", "iid"]}[stage]  # 'dirichlet' = 2B A1; 'compression' = Phase 2D
    for seed in cfg["seeds"]["model_seeds"]:
        for fold in cfg["outer_split"]["folds"]:
            for method, m in cfg["methods"].items():
                if stage == "dirichlet" and m["procedure"] == "centralized":
                    continue  # partition-independent; reused from the core stage
                for part in (["none"] if m["procedure"] == "centralized" else parts):
                    out.append((method, part, fold, seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="core")
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2b" / "core.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out_root = ROOT / "results" / {"compression": "phase2d", "structure": "phase2e", "adaptive": "phase2f"}.get(a.stage, "phase2b") / "runs"
    todo = jobs(cfg, a.stage)
    print(f"{len(todo)} jobs in stage {a.stage}", flush=True)
    if a.dry_run:
        return
    for i, (method, part, fold, seed) in enumerate(todo, 1):
        jid = job_id(method, part, fold, seed)
        if (out_root / jid / "result.json").exists():
            continue
        free = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
        if free < 8:
            print(f"STOP: free disk {free:.1f} GiB < 8", flush=True)
            sys.exit(3)
        t0 = time.time()
        os.makedirs(out_root / jid, exist_ok=True)
        with open(out_root / jid / "stdout.txt", "w") as so:
            rc = subprocess.run([sys.executable, "-I", str(ROOT / "src" / "fedgrande" / "phase2b" / "run_job.py"), "--method", method,
                                 "--partition", part, "--fold", str(fold), "--seed", str(seed), "--config", a.config,
                                 "--out-root", str(out_root)],
                                stdout=so, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        last = open(out_root / jid / "stdout.txt").read().strip().splitlines()[-1:] if rc == 0 else ["FAILED"]
        print(f"[{i}/{len(todo)}] {jid} rc={rc} {time.time()-t0:.0f}s free={free:.1f}GiB :: {last}", flush=True)
        if rc != 0:
            print("JOB FAILED -- see stdout.txt", flush=True)


if __name__ == "__main__":
    main()
