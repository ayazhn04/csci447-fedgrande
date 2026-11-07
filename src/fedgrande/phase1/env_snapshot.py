"""Phase 1: write a reproducible environment snapshot.

Package versions come from importlib.metadata (installed distribution metadata),
never from a module's __version__ attribute.
"""
import argparse
import datetime
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys

KEY_PACKAGES = [
    "GRANDE", "torch", "numpy", "pandas", "scipy", "scikit-learn", "category-encoders",
    "openml", "autogluon.features", "autogluon.core", "rtdl-num-embeddings",
    "xgboost", "catboost", "pip", "setuptools",
]


def git(worktree, *args):
    out = subprocess.run(["git", "-C", worktree, *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def version_or_none(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--grande-worktree", required=True)
    ap.add_argument("--expected-commit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    head = git(a.grande_worktree, "rev-parse", "HEAD")
    tracked_changes = git(a.grande_worktree, "status", "--porcelain", "--untracked-files=no")
    untracked = git(a.grande_worktree, "status", "--porcelain", "--untracked-files=all")

    dists = {}
    for d in md.distributions():
        name = d.metadata["Name"] if d.metadata else None
        if name:
            dists[name] = d.version

    grande_location = None
    try:
        grande_location = md.distribution("GRANDE").read_text("direct_url.json")
    except md.PackageNotFoundError:
        pass

    # Document the known category-encoders discrepancy explicitly (2.9.0 wheel reports __version__ '2.8.1').
    ce_dunder = None
    try:
        import category_encoders
        ce_dunder = getattr(category_encoders, "__version__", None)
    except Exception as exc:  # noqa: BLE001
        ce_dunder = f"import failed: {exc!r}"

    snap = {
        "label": a.label,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "grande": {
            "worktree": os.path.abspath(a.grande_worktree),
            "expected_commit": a.expected_commit,
            "head_commit": head,
            "commit_matches": head == a.expected_commit,
            "tracked_changes": tracked_changes.splitlines(),
            "untracked_files": [l for l in untracked.splitlines() if l.startswith("??")],
            "installed_distribution_version": version_or_none("GRANDE"),
            "direct_url_json": grande_location,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "full": sys.version,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "mac_ver": platform.mac_ver()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "conda": {
            "env_name": os.environ.get("CONDA_DEFAULT_ENV"),
            "prefix": os.environ.get("CONDA_PREFIX"),
        },
        "key_packages_metadata": {n: version_or_none(n) for n in KEY_PACKAGES},
        "category_encoders_module_dunder_version": ce_dunder,
        "all_distributions_metadata": dict(sorted(dists.items(), key=lambda kv: kv[0].lower())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(snap, f, indent=2)

    print(f"snapshot written: {a.out}")
    print(f"GRANDE head {head}  matches expected: {head == a.expected_commit}")
    print(f"tracked changes in worktree: {len(snap['grande']['tracked_changes'])}")
    for n, v in snap["key_packages_metadata"].items():
        print(f"  {n:22s} {v}")
    print(f"  category_encoders.__version__ (module attribute, NOT authoritative): {ce_dunder}")
    if head != a.expected_commit or tracked_changes:
        print("ERROR: commit mismatch or tracked modifications in GRANDE worktree", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
