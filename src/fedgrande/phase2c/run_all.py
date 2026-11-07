"""Run all 15 Phase 2C (fold, seed) jobs serially in subprocesses (resumable)."""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from fedgrande.paths import ROOT  # noqa: E402

cfg = json.load(open(ROOT / "configs" / "phase2c" / "personalization.json"))
out = ROOT / "results" / "phase2c" / "runs"
for seed in cfg["seeds"]:
    for fold in cfg["folds"]:
        d = out / f"f{fold}_s{seed}"
        if (d / "result.json").exists():
            continue
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        with open(d / "stdout.txt", "w") as so:
            rc = subprocess.run([sys.executable, "-I", str(ROOT / "src/fedgrande/phase2c/run_personalize.py"), "--fold", str(fold), "--seed", str(seed)],
                                stdout=so, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print(f"f{fold}_s{seed} rc={rc} {time.time()-t0:.0f}s", flush=True)
