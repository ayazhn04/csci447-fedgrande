# Phase 2D protocol: communication-efficient Federated GRANDE (frozen before the final runs)

**Status:** frozen. The frozen commit is the commit that adds this file. The configuration is
`configs/phase2d/compression.json`. This is the final algorithmic extension.

## Question

How much can FedAvg-GRANDE communication be reduced, by a smaller ensemble and/or INT8 client updates, without materially
degrading predictive performance?

## Unchanged from Phase 2B

Everything below is identical to Phase 2B:
- preprocessing (502 features) and the C2 outer folds with inner validation;
- the natural and IID partitions, and seeds 42, 447 and 2026;
- FedAvg with local Adam, 1 local epoch, full participation, at most 100 rounds;
- selection on validation AUC (patience 10, warm-up 10);
- thresholds, metrics, pooled-OOF aggregation, and 6 threads at "highest" float32 matmul precision.

The code path is `src/fedgrande/phase2b/run_job.py` → `fl/train.fedavg`.

## Configurations

| | Trees | Depth | Uplink | Trainable params | Source |
|---|---|---|---|---|---|
| **C0** | 256 | 5 | FP32 | 809,984 | **Phase 2B results reused** (`results/phase2b/runs/fedavg_grande__*`) |
| **C1** | 64 | 5 | FP32 | 202,496 | new |
| **C2** | 256 | 5 | INT8 client updates | 809,984 | new |
| **C3** | 64 | 5 | INT8 client updates | 202,496 | new |

`selected_variables` is 0.8 in every configuration, so V = 50.

**C0 engineering consistency check.** After adding the INT8 option, the C0 job for natural, fold 0, seed 42 was rerun. Its
test predictions and per-round validation AUC were **bitwise identical** to Phase 2B
(`results/phase2d/consistency/`). So the FP32 path is unchanged, and C0 is reused rather than rerun.

## INT8 uplink algorithm (`fl/compress.py`)

1. Clients train in FP32, exactly as in C0.
2. For every trainable tensor, compute Δ = θ_client − θ_global. The four tensors are `split_values`, `split_index_array`,
   `estimator_weights` and `leaf_classes_array`.
3. Quantize each tensor independently, with symmetric per-tensor quantization:
   - s = max|Δ| / 127;
   - q = clip(round(Δ / s), −127, 127), stored as int8. `torch.round` rounds half to even, which is deterministic.
   - If Δ is all zeros: s = 0 and q = 0, with no division.
4. **Sent to the server:** q (int8) plus s (float32) per tensor. Shapes are fixed by the shared architecture and are not
   transmitted.
5. **Server reconstruction:** Δ̂ = s·q in FP32, then θ̂_client = θ_global + Δ̂, followed by the usual
   training-row-weighted FedAvg over the θ̂_client.
6. **Not quantized:** the downlink (the FP32 global model), local and server model state, and structural buffers.
7. **Not used:** error feedback, stochastic rounding, sparsification or entropy coding.
8. **Diagnostics, per round.** Relative L2 error ‖Δ̂ − Δ‖ / ‖Δ‖ for each tensor: the mean and maximum over clients, plus
   the same over all tensors. Only these aggregated values are saved, not raw updates.

## Communication accounting: conceptual tensor communication bytes (`comm.accounting_v2`)

No serialization or network-protocol overhead is included. All configurations use the same definitions:

- **Downlink, per client per round:** the FP32 trainable global parameters, 4 bytes × params.
- **Uplink, per client per round:**
  - FP32: the updated parameters, 4 bytes × params.
  - INT8: 1 byte × params, plus 4 bytes × 4 tensors for the scales.
- **Once at initialization (downlink):** `features_by_estimator` (int64, E×V) for each client. The path and node index
  lists are regenerated from depth and cost 0 bytes. Buffers are never sent again.
- **Reported:** per client per round; aggregate per round (21 clients, full participation); cumulative to the best
  validation round and to the final round; downlink, uplink and total separately; and the compression ratio relative to C0.
- **C0's communication is recomputed** with the same function from its Phase 2B parameter shapes and round counts.

## Benchmark (pre-freeze, fold 0, seed 42, all 6 new jobs; runtime and validation only)

- **Wall time:** C1 54 s / 91 s; C2 327 s / 337 s; C3 62 s / 76 s (natural / IID).
- **Block total:** 947 s, which projects to **about 3.95 h** for the 90 jobs (up to about 4.4 h if more runs use all
  100 rounds).
- **Memory:** peak RSS at most 2.7 GB.
- **INT8 error:** maximum relative L2 quantization error per round 0.4–0.5%.
- These benchmark runs are development artifacts (`results/phase2d/benchmark/`). The final matrix reruns those jobs.

## Metrics and reporting

- Pooled-OOF ROC-AUC (primary), macro-F1 at 0.5, macro-F1 at the validation threshold, accuracy and confusion matrices.
- Per seed, then mean ± SD over the 3 seeds.
- ΔAUC = AUC_c − AUC_C0 and AUC retention = 100·AUC_c / AUC_C0, both paired by partition and seed.
- Paired fold/seed test-AUC differences against C0, descriptive only. No significance tests.
- Also: best and executed rounds, runtime, peak RSS, the INT8 error diagnostics, and a small structural sanity summary
  (consensus and unanimous fraction at the best round, mean update cosine up to the best round).
- **Figures:** AUC against cumulative GB to the best round (log x-axis), and validation AUC against cumulative GB per round.

## Outcome categories (fixed in advance)

- **Strong success:** a substantial cut in total communication while losing at most about 0.005–0.01 absolute AUC.
- **Moderate tradeoff:** a substantial cut with a modest loss.
- **Failure:** substantial degradation, with the failing component identified.

Readings:
- C1 good: ensemble redundancy is the main opportunity.
- C2 good: low-precision updates work.
- C3 good: the two are complementary.
- C2 fails while C1 succeeds: model-size reduction beats naive quantization.

## Excluded

Differential privacy, secure aggregation, top-k sparsification, error feedback, stochastic quantization, FedProx, FedAdam,
SCAFFOLD, new partitions or α values, personalization, HPO, quantization-aware training, 4-bit quantization, and
wave-held-out evaluation.

## Amendments

None at freeze time.
