"""Gate B: reconstruct dyads from the checksummed SpeedDating ARFF and save them with a report."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from fedgrande import dyads  # noqa: E402
from fedgrande.data import load_full  # noqa: E402
from fedgrande.paths import ARFF, DERIVED, RESULTS  # noqa: E402
from fedgrande.provenance import run_metadata, sha256_file, write_json  # noqa: E402


def main():
    full = load_full(verify=True)
    rows, report = dyads.reconstruct(full)
    os.makedirs(DERIVED, exist_ok=True)
    out_csv = DERIVED / "dyads_v1.csv"
    rows.to_csv(out_csv, index=False)
    report["outputs"] = {"dyads_csv": str(out_csv.relative_to(DERIVED.parents[2])), "dyads_csv_sha256": sha256_file(out_csv)}
    report["input_arff_sha256"] = sha256_file(ARFF)
    report["meta"] = run_metadata()
    write_json(RESULTS / "gate_b" / "dyads_report.json", report)
    print({k: report[k] for k in ("n_rows", "n_groups", "status_rows", "status_groups", "paired_rows_total", "dyads_size2")})
    print(report["validation_pairs"], report["validation_groups"])
    print("one-way:", [(o["row"], o["proposed_mirror"], o["values_row"], o["values_mirror_self_side"]) for o in report["one_way_pairs"]])
    print("ambiguous:", [(a["rows"], a["match_values"]) for a in report["ambiguous_groups"]])
    print("singletons:", [(s["row"], s["wave"], s["partner_side_missing"]) for s in report["singletons"]])


if __name__ == "__main__":
    main()
