"""Communication accounting from the actual tensors exchanged.

Per round, the server sends every participating client the global TRAINABLE parameters and receives the client's
updated trainable parameters (float32). Non-trainable GRANDE buffers:
  * features_by_estimator: identical for all clients; conceptually transmitted ONCE at initialization (counted once);
  * path_identifier_list / internal_node_index_list: pure functions of depth, regenerated locally (0 bytes).
"""
from __future__ import annotations


def tensor_bytes(state: dict) -> int:
    return int(sum(t.numel() * t.element_size() for t in state.values()))


def accounting(trainable: dict, buffers_sent_once: dict, n_clients: int, rounds_executed: int, best_round: int | None) -> dict:
    p = tensor_bytes(trainable)
    once = tensor_bytes(buffers_sent_once) * n_clients
    per_round_down, per_round_up = p * n_clients, p * n_clients
    per_round = per_round_down + per_round_up
    return {"trainable_parameters": int(sum(t.numel() for t in trainable.values())), "payload_bytes_per_model": p,
            "bytes_once_at_init_buffers": once, "bytes_server_to_clients_per_round": per_round_down,
            "bytes_clients_to_server_per_round": per_round_up, "bytes_per_round_total": per_round,
            "cumulative_bytes_to_best_round": once + per_round * int(best_round or 0),
            "cumulative_bytes_to_final_round": once + per_round * int(rounds_executed),
            "buffers_policy": "features_by_estimator sent once at init; structural index lists regenerated from depth"}


def accounting_v2(trainable: dict, buffers_sent_once: dict, n_clients: int, rounds_executed: int, best_round, uplink: str = "fp32") -> dict:
    """Phase 2D conceptual tensor communication bytes, split into downlink and uplink.

    downlink per client per round: FP32 global trainable parameters.
    uplink per client per round:   FP32 updated parameters (fp32) | INT8 deltas + one FP32 scale per tensor (int8).
    once at initialization:        features_by_estimator (int64) per client (downlink). Index lists: 0 bytes (regenerated).
    """
    from .compress import payload_bytes

    down = tensor_bytes(trainable)
    up = down if uplink == "fp32" else payload_bytes(trainable)
    once = tensor_bytes(buffers_sent_once) * n_clients
    per_round_down, per_round_up = down * n_clients, up * n_clients
    best = int(best_round or 0)
    return {"uplink": uplink, "trainable_parameters": int(sum(t.numel() for t in trainable.values())), "n_tensors": len(trainable),
            "downlink_bytes_per_client_per_round": down, "uplink_bytes_per_client_per_round": up,
            "bytes_once_at_init_buffers": once, "downlink_bytes_per_round": per_round_down, "uplink_bytes_per_round": per_round_up,
            "total_bytes_per_round": per_round_down + per_round_up,
            "cum_downlink_to_best": once + per_round_down * best, "cum_uplink_to_best": per_round_up * best,
            "cum_total_to_best": once + (per_round_down + per_round_up) * best,
            "cum_downlink_to_final": once + per_round_down * int(rounds_executed), "cum_uplink_to_final": per_round_up * int(rounds_executed),
            "cum_total_to_final": once + (per_round_down + per_round_up) * int(rounds_executed),
            "definition": "conceptual tensor communication bytes (no serialization/protocol overhead)"}
