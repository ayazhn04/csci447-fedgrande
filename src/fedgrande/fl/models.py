"""Model factories for Phase 2B: GRANDE 1.0.0 network (db79c860) and a fixed MLP."""
from __future__ import annotations

import torch
import torch.nn as nn

GRANDE_LR = {"split_values": 0.01, "split_index_array": 0.01, "estimator_weights": 0.005, "leaf_classes_array": 0.01}
GRANDE_BETAS = (0.9, 0.95)  # GRANDE 1.0.0 optimizer betas


def build_grande(n_features: int, seed: int, n_estimators: int = 256, depth: int = 5, selected_variables: float = 0.8):
    from GRANDE.model import GRANDE_Module  # pinned db79c860, used unmodified

    return GRANDE_Module(number_of_variables=n_features, number_of_classes=1, objective="binary", device="cpu",
                         depth=depth, n_estimators=n_estimators, dropout=0.0, selected_variables=selected_variables,
                         data_subset_fraction=1.0, bootstrap=False, missing_values=False, batch_size=256,
                         random_seed=seed, verbose=0)


def grande_optimizer(model):
    return torch.optim.Adam([{"params": getattr(model, n), "lr": lr, "betas": GRANDE_BETAS} for n, lr in GRANDE_LR.items()])


class MLP(nn.Module):
    """Fixed MLP: in -> 128 -> ReLU -> 64 -> ReLU -> 2 logits. No dropout, no batch norm (FedAvg-safe)."""

    def __init__(self, n_features: int, hidden=(128, 64)):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


MLP_LR = 1e-3


def build_mlp(n_features: int, seed: int):
    torch.manual_seed(seed)
    return MLP(n_features)


def mlp_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=MLP_LR)


FACTORY = {"grande": (build_grande, grande_optimizer), "mlp": (build_mlp, mlp_optimizer)}


def trainable_state(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def buffer_state(model) -> dict:
    return {n: b.detach().clone() for n, b in model.named_buffers()}


def load_trainable(model, state: dict):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(state[n])


def n_trainable(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
