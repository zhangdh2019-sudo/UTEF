# Bitcoin Paper Summary (Tuned Ours)

## Bitcoin-OTC
- Ours: AUC-PR 0.9145, MCC 0.1596, AURC(conf) 0.1532, ECE 0.0496
- Best values in table: AUC-PR 0.9145, MCC 0.1934, AURC(conf) 0.1143, ECE 0.0496
- vs TrustGuard-like: +0.0227 AUC-PR, +0.0933 MCC, 0.0495 lower AURC(conf)
- vs GAtrust-like: +0.0182 AUC-PR, +0.0936 MCC, 0.0504 lower AURC(conf)

## Bitcoin-Alpha
- Ours: AUC-PR 0.8706, MCC 0.0709, AURC(conf) 0.2135, ECE 0.0896
- Best values in table: AUC-PR 0.8723, MCC 0.0779, AURC(conf) 0.1536, ECE 0.0772
- vs TrustGuard-like: +0.0401 AUC-PR, +0.1492 MCC, 0.2334 lower AURC(conf)
- vs GAtrust-like: +0.0384 AUC-PR, +0.1224 MCC, 0.1411 lower AURC(conf)

## Claim Boundary
- Strong: Ours shows robust gains over trust-specific baselines (TrustGuard-like, GAtrust-like) on AUC-PR and MCC in both Bitcoin datasets.
- Moderate: Ours is top or near-top on AUC-PR, but not uniformly best on every uncertainty/calibration metric against all generic GNN baselines.
- Avoid: claiming universal best performance across all metrics.
