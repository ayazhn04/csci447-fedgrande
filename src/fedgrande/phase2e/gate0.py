"""Phase 2E Gate 0: exact trainable-tensor composition of the 256-tree GRANDE (502 inputs) and per-round savings."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from fedgrande.fl import compress  # noqa: E402
from fedgrande.fl.comm import tensor_bytes  # noqa: E402
from fedgrande.fl.models import build_grande, trainable_state  # noqa: E402
from fedgrande.paths import ROOT  # noqa: E402
from fedgrande.provenance import write_json  # noqa: E402

STRUCT_F1 = ["split_index_array"]
STRUCT_F2 = ["split_index_array", "split_values"]
K = 21


def main():
    st = trainable_state(build_grande(502, 42, n_estimators=256, depth=5))
    P = sum(t.numel() for t in st.values())
    B = tensor_bytes(st)
    rows = [{"tensor": k, "shape": list(t.shape), "params": t.numel(), "fp32_bytes": t.numel() * 4,
             "pct_params": 100 * t.numel() / P, "pct_fp32_payload": 100 * t.numel() * 4 / B} for k, t in st.items()]
    full_round = 2 * K * B                                    # FP32 down + FP32 up, 21 clients

    def post(frozen, int8=False):
        act = {k: v for k, v in st.items() if k not in frozen}
        down = tensor_bytes(act) * K
        up = (compress.payload_bytes(act) if int8 else tensor_bytes(act)) * K
        return {"downlink_per_round": down, "uplink_per_round": up, "total_per_round": down + up,
                "reduction_vs_F0_per_round_pct": 100 * (1 - (down + up) / full_round)}
    strategies = {"F0": post([]), "F1": post(STRUCT_F1), "F2": post(STRUCT_F2), "F3": post(STRUCT_F2, int8=True)}
    gate = strategies["F2"]["reduction_vs_F0_per_round_pct"] >= 10
    out = {"model": "GRANDE 256 estimators, depth 5, 502 inputs, V=50", "total_params": P, "total_fp32_bytes": B,
           "tensors": rows, "F0_bytes_per_round": full_round, "post_freeze_per_round": strategies,
           "gate0_threshold_pct": 10, "gate0_pass": bool(gate),
           "note": "per-round post-freeze steady state (rounds >= 22); round 21 additionally broadcasts the frozen round-20 structure once"}
    write_json(ROOT / "results" / "phase2e" / "gate0.json", out)
    for r in rows:
        print(f"{r['tensor']:20s} {str(r['shape']):16s} {r['params']:8d} {r['fp32_bytes']:9d} {r['pct_params']:6.2f}% {r['pct_fp32_payload']:6.2f}%")
    print("total", P, B)
    for k, v in strategies.items():
        print(k, {kk: (round(vv / 1e6, 3) if "pct" not in kk else round(vv, 2)) for kk, vv in v.items()})
    print("GATE 0 PASS:", gate)


if __name__ == "__main__":
    main()
