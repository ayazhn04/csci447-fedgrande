# Start here

This page is for a teammate who has not followed any of the development conversations. It uses no phase names until it
explains them. Reading time is about 10 minutes.

## 1. What was the original assignment?

- **The course.** CSCI447 Machine Learning (project CSCI447P016) asks for a research project built around a published
  ML method.
- **Our method.** **GRANDE**, a gradient-based decision-tree ensemble for tabular data.
- **Checkpoint 1** (archived in `archive/checkpoint1/`). We picked a dataset, ran GRANDE, and mapped the paper's
  equations to its code.
- **The rest of the project.** We turned this into a study of training GRANDE in a *federated* setting.
- **To confirm.** The exact assignment wording is in the course materials, not in this repository; please check the
  final-submission requirements there.

## 2. What is GRANDE?

GRANDE (Marton et al.) is an ensemble of **hard, axis-aligned decision trees** whose parameters are learned **by gradient
descent**.

Each tree of depth 5 has 31 internal nodes and 32 leaves. Its parameters are:
- **split feature logits** (`split_index_array`): which feature each node tests, chosen by argmax among 50 candidate
  features assigned to that tree;
- **split thresholds** (`split_values`);
- **leaf values** (`leaf_classes_array`);
- **per-tree weights** (`estimator_weights`).

A straight-through trick makes the hard splits differentiable. We use 256 trees, which gives **809,984 trainable
parameters**.

## 3. Why federated learning?

- **No gradient boosting needed.** Usual tree ensembles (gradient boosting) are hard to train federatedly. GRANDE's
  trees are trained by gradients, so the standard federated algorithm **FedAvg** (each client trains locally; a server
  averages parameters) applies directly.
- **Natural clients.** Our dataset comes with a natural split into **21 separate events ("waves")**. That makes a
  realistic setting of 21 data holders who cannot pool their data.

## 4. What dataset?

- **SpeedDating** (OpenML 40536): 8,378 rows, 16.5% positive.
- **Target:** `match`, whether both people wanted to meet again.
- **Features:** 119 raw features. We removed `decision` and `decision_o` because together they define the label, and
  `wave` because it is the client identifier.
- **Encoding:** after label-free encoding, every model sees 502 numeric inputs.

## 5. What problem did we find with the historical evaluation?

- **Mirror rows.** Each date appears **twice**, once from each participant's side. The two rows mirror each other and
  share the label.
- **The leak.** The ordinary row-wise 5-fold CV we used at Checkpoint 1 put **79.8% of test rows' mirror twins in the
  training set**.
- **The fix.** We grouped mirror rows into **leakage groups** (4,175 two-row dyads, 4 ambiguous groups, 10 singletons).
  Every evaluation since uses **grouped folds**, so a group never spans training and test.
- **Caution.** We showed the split was structurally leaky. We did *not* show that any specific old number was inflated.

## 6. What did Phase 2A establish? (foundations)

- The new GRANDE 1.0.0 behaves **bitwise identically** to the historical version we had used.
- Mirror-row leakage is real. The grouped split removes it.
- The 21 waves differ a lot: sizes 50–968 rows; positive rate 8–31%.
- Runtimes were measured and used to choose a 256-tree model.

## 7. What did Phase 2B establish? (the core federated study)

We compared five models on the 21 natural clients and on a **size-matched IID control** (same client sizes, shuffled
waves):
- centralized, local-only and FedAvg GRANDE;
- centralized and FedAvg MLP.

| Model | AUC |
|---|---|
| Centralized GRANDE | **0.857** |
| FedAvg GRANDE | **0.844** (natural), 0.845 (IID) |
| Local-only GRANDE | 0.741 |
| FedAvg MLP | 0.850 |

- **Federation works**, and natural ≈ IID: wave heterogeneity hardly matters here.
- An added **extreme label-skew stress test** (Dirichlet α = 0.1) hurt much more. It mainly broke *calibration*.
- Clients' trees **agree on split features** more and more: mean consensus about 0.98 by round 20.

## 8. Why did we run Phase 2C? (calibration vs personalization)

- **The question.** Under extreme label skew, should each client personalize parts of the trees (leaves, tree weights),
  or does a one-number correction per client (an intercept) suffice?
- **The answer.** **The intercept is enough.** Personalizing the trees did not improve ranking within clients, and was
  worse calibrated.

## 9. Why did we run Phase 2D? (static communication compression)

- **The problem.** FedAvg GRANDE sends **136 MB per round** (about 11 GB until it converges).
- **What we tried.** Fewer trees (64) and 8-bit (INT8) client updates.
- **Results.** INT8 is free (1.5× less); 64 trees saves about 5× for about 0.01 AUC; both together save about 8×.

## 10. Why did we run Phase 2E? (freeze the tree structure)

- **Two observations:**
  1. the split features and thresholds are **98% of the parameters**;
  2. clients agree on them by about round 20.
- **The idea.** Stop federating structure after round 20, and keep training only the leaves and tree weights.
- **The result.** **AUC dropped by only about 0.002 while communication fell about 4×.** Adding INT8 on top helps a
  little more.

## 11. Why did we run Phase 2F? (let consensus choose when to freeze)

- **The question.** Round 20 was picked by hand. Can the agreement signal choose the moment automatically?
- **G1 (global rule).** "Freeze everything once agreement is very high for 3 rounds." It froze at **round 20.4 ± 1.4**
  on average and matched the hand-picked schedule exactly.
- **G2 (per-node rule).** Freezes node by node. It cost about 20% more communication and was less stable.

## 12. What is the final scientific conclusion?

> FedAvg can train GRANDE across real clients at about 98.5% of centralized AUC. Natural client heterogeneity is benign
> here; extreme label skew mainly breaks calibration, which a per-client intercept fixes. GRANDE's communication is
> dominated by tree structure, which stabilizes early. Freezing it after it stabilizes cuts communication about 4× at
> negligible cost. **A consensus-gated rule reproduces that freeze point automatically but does not improve on it.**

## 13. Which results matter most for the report?

1. The leakage-aware grouped evaluation (§5).
2. Centralized vs FedAvg vs local (`fig02`).
3. Natural vs IID vs label skew, and structural consensus (`fig03`).
4. Calibration beats personalization (`fig05`).
5. The communication frontier: compression, fixed freeze, adaptive freeze (`fig08`).
6. G1 rediscovers round about 20 (`fig11`).

## 14. What to read next

- [RESULTS_LEDGER.md](RESULTS_LEDGER.md): exact numbers.
- [FIGURE_CATALOG.md](FIGURE_CATALOG.md): which figures to use.
- [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md): the seven questions, each with hypothesis, result and limitation.
- [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md): what we can and cannot say.
- [TEAMMATE_ONBOARDING.md](TEAMMATE_ONBOARDING.md): setup and workflow.
