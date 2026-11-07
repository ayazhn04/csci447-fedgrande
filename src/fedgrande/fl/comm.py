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
