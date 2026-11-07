"""Gate E/F driver: run every spec of configs/phase2/gate_ef_runs.json in its own subprocess (fresh peak-RSS).

Skips runs whose metrics.json already exists (resumable). Usage: python gate_ef_run.py [run_id ...]
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from fedgrande.paths import CONFIGS, ROOT, RESULTS  # noqa: E402


def main():
    cfg = json.load(open(CONFIGS / "gate_ef_runs.json"))
    only = set(sys.argv[1:])
    for r in cfg["runs"]:
        if only and r["run_id"] not in only:
            continue
        outdir = RESULTS / "gate_ef" / r["run_id"]
        if (outdir / "metrics.json").exists():
            print("skip (exists)", r["run_id"], flush=True)
            continue
        spec = {"run_id": r["run_id"], "label": r.get("label", "runtime benchmark (GRANDE 1.0.0)"), "scheme": r["scheme"],
                "outer_fold": r["outer_fold"], "estimator_params": {**cfg["common_estimator_params"], **r["overrides"]}}
        os.makedirs(outdir, exist_ok=True)
        spec_path = outdir / "spec.json"
        json.dump(spec, open(spec_path, "w"), indent=2)
        with open(outdir / "stdout.txt", "w") as so:
            rc = subprocess.run([sys.executable, "-I", str(ROOT / "src" / "fedgrande" / "grande_runner.py"), "--spec", str(spec_path),
                                 "--outdir", str(outdir)], stdout=so, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        tail = open(outdir / "stdout.txt").read().strip().splitlines()[-1:]
        print(f"{r['run_id']}: exit {rc} :: {tail}", flush=True)


if __name__ == "__main__":
    main()
