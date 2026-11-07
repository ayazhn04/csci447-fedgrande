"""Construct the core GRANDE network of either pinned commit with explicit, identical arguments.

The two commits ship the same-named package (``GRANDE``), so each implementation is used in its own conda
environment / process. Nothing in the third-party source is modified.

Parameter/state mapping (verified by E0a, identical names and shapes in both commits):

    split_values          [E, I, V]  parameter   split thresholds
    split_index_array     [E, I, V]  parameter   split-feature logits (softmax + straight-through argmax in both)
    estimator_weights     [E, L]     parameter   instance-wise estimator weight logits
    leaf_classes_array    [E, L]     parameter   leaf values (binary/regression)
    features_by_estimator [E, V]     buffer      feature subset of each estimator (non-trainable)
    path_identifier_list  [L, D]     buffer      structural constant (left/right along each path)
    internal_node_index_list [L, D]  buffer      structural constant (internal node along each path)

with E = n_estimators, I = 2**depth - 1, L = 2**depth, D = depth, V = effective selected variables.
"""
from __future__ import annotations

PARAMETERS = ["split_values", "split_index_array", "estimator_weights", "leaf_classes_array"]
BUFFERS = ["features_by_estimator", "path_identifier_list", "internal_node_index_list"]
STATE_MAPPING = {name: name for name in PARAMETERS + BUFFERS}  # ref name -> dev name


def build_module(impl: str, *, number_of_variables, depth, n_estimators, selected_variables, missing_values,
                 random_seed, dropout=0.0, data_subset_fraction=1.0, bootstrap=False, batch_size=256,
                 objective="binary", number_of_classes=1, device="cpu"):
    if impl == "ref":
        from GRANDE.GRANDE import GRANDE_Module  # 07f7278b: params-dict constructor

        params = dict(number_of_variables=number_of_variables, number_of_classes=number_of_classes, objective=objective,
                      device=device, depth=depth, n_estimators=n_estimators, dropout=dropout,
                      selected_variables=selected_variables, data_subset_fraction=data_subset_fraction,
                      bootstrap=bootstrap, missing_values=missing_values, batch_size=batch_size,
                      random_seed=random_seed, verbose=0)
        return GRANDE_Module(params=params)
    if impl == "dev":
        from GRANDE.model import GRANDE_Module  # db79c860: keyword constructor

        return GRANDE_Module(number_of_variables=number_of_variables, number_of_classes=number_of_classes,
                             objective=objective, device=device, depth=depth, n_estimators=n_estimators,
                             dropout=dropout, selected_variables=selected_variables,
                             data_subset_fraction=data_subset_fraction, bootstrap=bootstrap,
                             missing_values=missing_values, batch_size=batch_size, random_seed=random_seed, verbose=0)
    raise ValueError(impl)


def count_trainable(module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
