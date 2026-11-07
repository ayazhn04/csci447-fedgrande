"""Phase 2D: deterministic symmetric per-tensor INT8 quantization of client UPDATES (uplink only).

s = max|d| / 127 ; q = clip(round(d / s), -127, 127) as int8 ; d_hat = s * q (float32).
All-zero tensor: s = 0 and q = 0 (no division), reconstructed exactly as zeros.
torch.round uses round-half-to-even (deterministic). Payload per tensor: numel int8 bytes + one float32 scale (4 bytes);
shapes/dtypes are fixed by the shared architecture and need no transmission.
"""
from __future__ import annotations

import torch

QMAX = 127
SCALE_BYTES = 4


def quantize(d: torch.Tensor):
    d = d.detach().to(torch.float32)
    amax = float(d.abs().max()) if d.numel() else 0.0
    if amax == 0.0:
        return torch.zeros_like(d, dtype=torch.int8), torch.tensor(0.0, dtype=torch.float32)
    s = torch.tensor(amax / QMAX, dtype=torch.float32)
    q = torch.clamp(torch.round(d / s), -QMAX, QMAX).to(torch.int8)
    return q, s


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * s


def payload_bytes(state: dict) -> int:
    """INT8 uplink payload for a dict of trainable tensors: int8 values + one fp32 scale per tensor."""
    return int(sum(t.numel() * 1 + SCALE_BYTES for t in state.values()))


def roundtrip_update(local: dict, glob: dict):
    """Quantize local-global deltas per tensor and return (reconstructed local state, per-tensor relative L2 errors).

    Inputs are not modified (new tensors are created)."""
    out, rel = {}, {}
    for k in glob:
        d = local[k] - glob[k]
        q, s = quantize(d)
        dh = dequantize(q, s)
        out[k] = glob[k] + dh
        n = float(torch.linalg.vector_norm(d))
        rel[k] = float(torch.linalg.vector_norm(dh - d)) / n if n > 0 else 0.0
    return out, rel
