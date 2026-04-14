# Reviewer-Oriented Revision Guide

## What Was Fixed in Code

1. Multi-dataset protocol:
- Added dataset support in `main_link.py` via `--dataset`.
- Added loaders for:
  - `bitcoin_otc`
  - `bitcoin_alpha`
  - `wiki_rfa`
- All datasets follow the same temporal split rule: `train < val < test` by timestamp.

2. Coverage / selective prediction protocol:
- Added confidence-based selective ranking (`confidence_maxprob`, `confidence_margin`) in addition to uncertainty-based modes.
- Evaluator now exports:
  - primary selective curve (`risk_coverage`)
  - confidence selective curve (`risk_coverage_confidence`)
  - raw model-uncertainty curve (`risk_coverage_model_uncertainty`)

3. Reproducible experiment suite:
- `scripts/run_reviewer_suite.py` runs full comparison + ablation for all three datasets.
- `scripts/compile_reviewer_results.py` aggregates tables and computes paired exact sign-flip tests.

## Method Claims: Strong vs Weak (for paper writing)

Strong claims (safe):
- The model is evaluated under strict temporal leakage-free protocol.
- The framework supports both uncertainty-based and confidence-based decision filtering.
- On some datasets/metrics, the model improves ranking/calibration-selective trade-offs.

Weak claims (must avoid overstatement):
- Do not claim "best on all metrics/datasets".
- Do not claim uncertainty selective is always better than confidence selective.
- Do not claim strict SOTA unless baselines are official implementations and significance is robust.

## Recommended Experiment Section Structure

1. Protocol and fairness:
- Temporal split, train-only propagation graph, shared preprocessing.

2. Main comparison:
- 9 baselines (Node2Vec+MLP, GCN, GraphSAGE, GAT, GAtrust-like, TrustGuard-like, Guardian, MC Dropout, Ours).

3. Ablation:
- `ours`, `ours_nounc`, `ours_1hop`, `ours_notemp`.

4. Decision-oriented uncertainty:
- Report both confidence-based and uncertainty-based risk-coverage.
- Explicitly state where uncertainty ranking is stronger/weaker than confidence ranking.

5. Statistical validation:
- Paired exact sign-flip tests over seed-wise results.

