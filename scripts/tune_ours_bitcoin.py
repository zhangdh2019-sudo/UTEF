#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "tune_ours_bitcoin"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrialCfg:
    num_hops: int
    hidden_dim: int
    dropout: float
    lr: float
    lambda_uncertainty: float
    lambda_ranking: float
    u_alpha: float
    label_smoothing: float
    classification_loss: str
    focal_gamma: float
    selective_uncertainty: str
    selective_hybrid_weight: float

    def to_args(self) -> List[str]:
        return [
            "--num-hops",
            str(self.num_hops),
            "--hidden-dim",
            str(self.hidden_dim),
            "--dropout",
            str(self.dropout),
            "--lr",
            str(self.lr),
            "--lambda-uncertainty",
            str(self.lambda_uncertainty),
            "--lambda-ranking",
            str(self.lambda_ranking),
            "--u-alpha",
            str(self.u_alpha),
            "--label-smoothing",
            str(self.label_smoothing),
            "--classification-loss",
            self.classification_loss,
            "--focal-gamma",
            str(self.focal_gamma),
            "--selective-uncertainty",
            self.selective_uncertainty,
            "--selective-hybrid-weight",
            str(self.selective_hybrid_weight),
        ]


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def read_ours_row(agg_csv: Path) -> Dict[str, float]:
    rows = list(csv.DictReader(agg_csv.open("r", encoding="utf-8")))
    ours = [r for r in rows if r["model"] == "ours"]
    if not ours:
        raise RuntimeError(f"No ours row in {agg_csv}")
    r = ours[0]
    out = {k: float(v) for k, v in r.items() if k.endswith("_mean")}
    return out


def score_metrics(m: Dict[str, float]) -> float:
    # Focus on paper-useful metrics: AUC-PR, MCC, confidence-based AURC, ECE.
    return (
        4.0 * m["auc_pr_mean"]
        + 1.8 * m["mcc_mean"]
        - 1.6 * m["aurc_confidence_mean"]
        - 0.4 * m["ece_after_mean"]
    )


def random_trials(seed: int, n_trials: int) -> List[TrialCfg]:
    rng = random.Random(seed)
    grid = list(
        itertools.product(
            [2, 3],
            [64, 96],
            [0.05, 0.10, 0.20],
            [5e-4, 1e-3, 2e-3],
            [0.0, 0.02, 0.05, 0.08],
            [0.02, 0.05, 0.08, 0.12],
            [0.35, 0.5, 0.65, 0.8],
            [0.0, 0.02, 0.05],
            ["bce", "focal"],
            [1.5, 2.0],
            ["model", "calibrated_entropy", "rank_hybrid", "confidence_maxprob"],
            [0.45, 0.55, 0.65],
        )
    )
    rng.shuffle(grid)
    chosen = grid[:n_trials]
    trials: List[TrialCfg] = []
    for g in chosen:
        trials.append(
            TrialCfg(
                num_hops=g[0],
                hidden_dim=g[1],
                dropout=g[2],
                lr=g[3],
                lambda_uncertainty=g[4],
                lambda_ranking=g[5],
                u_alpha=g[6],
                label_smoothing=g[7],
                classification_loss=g[8],
                focal_gamma=g[9],
                selective_uncertainty=g[10],
                selective_hybrid_weight=g[11],
            )
        )
    return trials


def tune_dataset(dataset: str, trials: List[TrialCfg], seeds: int, epochs: int, patience: int) -> Dict[str, object]:
    best = None
    records: List[Dict[str, object]] = []

    for i, cfg in enumerate(trials, start=1):
        out_csv = OUT_DIR / f"{dataset}_trial_{i:03d}.csv"
        cmd = [
            sys.executable,
            "main_link.py",
            "--dataset",
            dataset,
            "--data-root",
            "data",
            "--feature-type",
            "node2vec",
            "--models",
            "ours",
            "--node2vec-epochs",
            "18",
            "--epochs",
            str(epochs),
            "--patience",
            str(patience),
            "--seeds",
            str(seeds),
            "--out-csv",
            str(out_csv),
        ] + cfg.to_args()

        run_cmd(cmd)
        agg = read_ours_row(out_csv.with_name(out_csv.stem + "_agg.csv"))
        s = score_metrics(agg)
        rec = {
            "dataset": dataset,
            "trial": i,
            "score": s,
            "cfg": cfg.__dict__,
            "metrics": agg,
        }
        records.append(rec)
        if best is None or s > best["score"]:
            best = rec

    assert best is not None
    out_json = OUT_DIR / f"{dataset}_tune_summary.json"
    out_json.write_text(json.dumps({"best": best, "records": records}, indent=2), encoding="utf-8")
    return best


def main() -> int:
    trials = random_trials(seed=20260412, n_trials=16)
    best_otc = tune_dataset("bitcoin_otc", trials, seeds=2, epochs=80, patience=30)
    best_alpha = tune_dataset("bitcoin_alpha", trials, seeds=2, epochs=80, patience=30)

    summary = {
        "bitcoin_otc": best_otc,
        "bitcoin_alpha": best_alpha,
    }
    (OUT_DIR / "best_configs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved best config summary: {OUT_DIR / 'best_configs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
