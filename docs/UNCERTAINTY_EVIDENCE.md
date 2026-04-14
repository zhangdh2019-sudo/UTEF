# Uncertainty evidence for selective prediction (paper-facing)

This repo evaluates **edge-level uncertainty** with metrics that reviewers typically expect for *selective prediction*, not necessarily *perfect probabilistic calibration*.

## What to claim (wording)

- Prefer: **“uncertainty is useful for selective prediction / risk-aware filtering”**.
- Avoid: **“well-calibrated uncertainty”** unless ECE is stable and strong across seeds (often hard on imbalanced link prediction).

## Selective uncertainty protocol (important)

Decisions use **temperature + bias**–scaled probabilities \(p_{\mathrm{cal}}\) (fitted on val NLL, **multi-restart LBFGS** on \(T,b\)).  
Sorting for risk–coverage / filtering uses **`rank_hybrid`** by default:

\[
u_{\mathrm{sel}} = w \cdot \mathrm{rank}(u_{\mathrm{model}}) + (1-w)\cdot \mathrm{rank}(H(p_{\mathrm{cal}}))
\]

This aligns selective prediction with **post-hoc calibration** (same \(p_{\mathrm{cal}}\) as the threshold) while retaining **model uncertainty**.  
CLI: `--selective-uncertainty {model,calibrated_entropy,rank_hybrid}`, `--selective-hybrid-weight` (default **0.52**).  
Ablations: report `model` vs `rank_hybrid` in appendix if needed.

## Metrics implemented

| Metric | Meaning |
|--------|---------|
| **Risk–coverage curve** | Sort test edges by **ascending** uncertainty; increase coverage by including more edges. **Risk** = 1 − accuracy at a fixed threshold (val-tuned, test-frozen). |
| **AURC / EAURC** | Discrete curve summary: mean risk over coverage grid; EAURC = AURC − overall risk (excess vs accepting all edges). *Lower AURC is better* if risk is defined as misclassification rate. |
| **Selective @coverage** | Accuracy when keeping the lowest-uncertainty **70% / 80% / 90%** (`sel_acc_70`, … in CSV). |
| **Uncertainty filtering** | Remove top **10% / 20% / 30%** highest-uncertainty edges; report accuracy / risk / retain fraction (`filt20_*` in CSV). |
| **Curve diagnostics** | `rc_violation_rate`: fraction of adjacent curve points where risk **decreases** when coverage increases (should be **small** if uncertainty orders errors toward the tail). `rc_spearman_cov_risk`: Spearman ρ(coverage, risk) (typically **positive**). |
| **ECE (secondary)** | **Temperature + bias** on **val** (fit `logit' = logit(p)/T + b` by NLL), then applied to **test** (can be noisy; keep as supplementary). |

## TrustGuard baseline note

`baselines/trustguard_like.py` is a **lightweight reimplementation inspired by** TrustGuard’s spatial–temporal signed GNN idea (IEEE TDSC’24; public code: `https://github.com/Jieerbobo/TrustGuard`). It is **not** a line-by-line reproduction of the official repository, but provides a **fair same-protocol** baseline under our train-only adjacency + temporal split.

## Commands

**Main comparison + uncertainty columns + JSON for plotting**

```bash
python main_link.py \
  --seeds 5 \
  --classification-loss bce \
  --models ours,gatrust,trustguard,guardian \
  --out-csv outputs/uncertainty_compare.csv \
  --export-uncertainty-json outputs/uncertainty_compare.json
```

**Ablations (ours family)**

```bash
python main_link.py \
  --seeds 5 \
  --classification-loss bce \
  --models ours,ours_nounc,ours_1hop,ours_notemp \
  --out-csv outputs/ablation_ours.csv \
  --export-uncertainty-json outputs/ablation_ours.json
```

- `ours_nounc`: `--lambda-uncertainty` forced to `0` during training.  
- `ours_1hop`: `num_hops=1`.  
- `ours_notemp`: temporal edge extras disabled (`use_temporal_features=False`).

## How to plot the risk–coverage curve

Use `outputs/*.json` → `seeds[*].models["ours"].risk_coverage.{coverage,risk}` (matplotlib/plotly).

## Interpreting “logical” results

- **Filtering / selective accuracy** should usually **improve** when dropping high-uncertainty edges, *unless* uncertainty is uninformative (then gains are small or inconsistent across seeds).
- **AURC** comparisons across models are only meaningful when **the same** threshold rule and **the same** risk definition are used (this codebase: val-tuned threshold + test risk at that threshold).
- **EAURC < −0.05** is a *strong* bar; on real graphs it may or may not hold—report mean±std honestly.
