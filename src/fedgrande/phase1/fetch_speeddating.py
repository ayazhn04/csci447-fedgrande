"""Phase 1: download OpenML 40536 metadata and the original ARFF; verify OpenML's MD5.

Standard library only. Never overwrites an existing ARFF; re-verifies it instead.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request

DATASET_ID = 40536
API = "https://www.openml.org/api/v1/json"


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def file_hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    desc = get_json(f"{API}/data/{DATASET_ID}")
    feats = get_json(f"{API}/data/features/{DATASET_ID}")
    quals = get_json(f"{API}/data/qualities/{DATASET_ID}")
    for name, obj in [("description", desc), ("features", feats), ("qualities", quals)]:
        with open(os.path.join(a.out_dir, f"openml_{DATASET_ID}_{name}.json"), "w") as f:
            json.dump(obj, f, indent=2)

    d = desc["data_set_description"]
    arff_path = os.path.join(a.out_dir, f"SpeedDating_openml{DATASET_ID}_v{d['version']}.arff")
    if os.path.exists(arff_path):
        print(f"ARFF already present, not overwriting: {arff_path}")
    else:
        print(f"downloading {d['url']}")
        urllib.request.urlretrieve(d["url"], arff_path)

    md5 = file_hash(arff_path, "md5")
    sha256 = file_hash(arff_path, "sha256")
    manifest = {
        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_id": int(d["id"]),
        "name": d["name"],
        "version": int(d["version"]),
        "upload_date": d["upload_date"],
        "licence": d.get("licence"),
        "default_target_attribute": d["default_target_attribute"],
        "ignore_attribute": d.get("ignore_attribute"),
        "file_id": d["file_id"],
        "arff_url": d["url"],
        "parquet_url": d.get("parquet_url"),
        "arff_file": os.path.basename(arff_path),
        "arff_bytes": os.path.getsize(arff_path),
        "md5_expected_openml": d.get("md5_checksum"),
        "md5_observed": md5,
        "md5_ok": md5 == d.get("md5_checksum"),
        "sha256_observed": sha256,
    }
    with open(os.path.join(a.out_dir, "download_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for k, v in manifest.items():
        print(f"{k:28s} {v}")
    if not manifest["md5_ok"]:
        print("ERROR: MD5 mismatch against OpenML checksum", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
