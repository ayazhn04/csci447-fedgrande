"""Project paths (all relative to the repository root, derived from this file's location)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "openml_40536"
ARFF = RAW_DIR / "SpeedDating_openml40536_v1.arff"
MANIFEST = RAW_DIR / "download_manifest.json"
DERIVED = ROOT / "data" / "derived" / "phase2"
RESULTS = ROOT / "results" / "phase2"
CONFIGS = ROOT / "configs" / "phase2"
THIRD_PARTY = ROOT / "third_party"
GRANDE_WORKTREES = {
    "ref": (THIRD_PARTY / "grande_ref_07f7278b", "07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb"),
    "dev": (THIRD_PARTY / "grande_dev_db79c860", "db79c86060f06c15f46c3bfb9207dc30c3cb91ae"),
}
