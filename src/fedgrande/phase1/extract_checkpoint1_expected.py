"""Phase 1: derive the historical expected values from the archived Checkpoint 1 answer text.

Parses archive/checkpoint1/checkpoint1_moodle_answers_from_chat_paste.txt so that no historical number has to
be typed by hand. Standard library only. Fails loudly if any expected section cannot be found.
"""
import argparse
import hashlib
import json
import re
import sys


def need(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        sys.exit(f"ERROR: pattern not found in archived answers: {pattern!r}")
    return m


def as_int(s):
    return int(s.replace(",", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-first-five-csv", required=True)
    a = ap.parse_args()

    raw = open(a.answers, "rb").read()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    q11 = text[need(r"Answer text Question 11", text).start():need(r"Answer files Question 11", text).start()]

    shape = need(r"X has shape \((\d+), (\d+)\) and the target y has shape \((\d+),\)\. After combining the "
                 r"predictors and target, the dataset has ([\d,]+) rows and (\d+) columns", q11)
    dataset = need(r"SpeedDating \(OpenML ID (\d+), version (\d+)\)", q11)

    marker = "The first five rows of the loaded predictors with match added as the final column are:"
    i = lines.index(marker)
    first_five = lines[i + 1:i + 7]
    if not first_five[0].startswith("has_null,") or len(first_five) != 6:
        sys.exit("ERROR: first-five-rows block not found where expected")

    c0 = need(r"class 0: ([\d,]+) samples", q11)
    c1 = need(r"class 1: ([\d,]+) samples", q11)
    minority = need(r"our computed value is ([\d.]+)%", q11)
    miss_tot = need(r"we found ([\d,]+) missing values across (\d+) predictor columns", q11)
    tgt_miss = need(r"The target match contains (\d+) missing values", q11)

    msec = q11[need(r"4\. Missing values", q11).end():need(r"We verified this directly", q11).start()]
    missing = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(\S+)\s+(\d+)\s*$", msec, re.M)}
    folds = [{"fold": int(m.group(1)), "train": int(m.group(2)), "test": int(m.group(3)), "overlap": int(m.group(4))}
             for m in re.finditer(r"fold (\d): train=(\d+), test=(\d+), train/test overlap=(\d+)", q11)]
    held = need(r"all ([\d,]+) samples are held out exactly once", q11)
    ce = need(r"category-encoders: ([\d.]+)", text)

    exp = {
        "source_file": a.answers,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "note": "Values parsed programmatically from the archived Checkpoint 1 answer text; nothing typed by hand.",
        "openml_id": int(dataset.group(1)),
        "openml_version": int(dataset.group(2)),
        "X_shape": [int(shape.group(1)), int(shape.group(2))],
        "y_shape": [int(shape.group(3))],
        "Xy_shape": [as_int(shape.group(4)), int(shape.group(5))],
        "header": first_five[0].split(","),
        "first_five_rows_csv_lines": first_five,
        "class_counts": {"0": as_int(c0.group(1)), "1": as_int(c1.group(1))},
        "minority_pct_reported": minority.group(1),
        "missing_total": as_int(miss_tot.group(1)),
        "missing_columns": int(miss_tot.group(2)),
        "target_missing": int(tgt_miss.group(1)),
        "missing_by_column": missing,
        "folds": folds,
        "held_out_exactly_once": as_int(held.group(1)),
        "category_encoders_moodle_q14": ce.group(1),
    }
    # Internal consistency of the archived text itself (reported, not corrected).
    exp["internal_consistency"] = {
        "missing_table_sum": sum(missing.values()),
        "missing_table_sum_equals_reported_total": sum(missing.values()) == exp["missing_total"],
        "missing_table_rows_equals_reported_columns": len(missing) == exp["missing_columns"],
        "header_length_equals_Xy_columns": len(exp["header"]) == exp["Xy_shape"][1],
    }
    with open(a.out_json, "w") as f:
        json.dump(exp, f, indent=2)
    with open(a.out_first_five_csv, "w", newline="") as f:
        f.write("\n".join(first_five) + "\n")
    print(json.dumps({k: v for k, v in exp.items() if k not in ("header", "first_five_rows_csv_lines",
                                                                 "missing_by_column")}, indent=2))
    if not all(exp["internal_consistency"].values()):
        print("WARNING: archived answer text is internally inconsistent (see internal_consistency)", file=sys.stderr)


if __name__ == "__main__":
    main()
