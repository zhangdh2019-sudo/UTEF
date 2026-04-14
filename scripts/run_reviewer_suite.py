#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"


@dataclass
class DatasetCfg:
    name: str
    epochs: int
    patience: int
    num_hops: int
    node2vec_epochs: int


DATASETS = [
    DatasetCfg("bitcoin_otc", epochs=90, patience=35, num_hops=3, node2vec_epochs=20),
    DatasetCfg("bitcoin_alpha", epochs=90, patience=35, num_hops=3, node2vec_epochs=20),
    DatasetCfg("wiki_rfa", epochs=60, patience=20, num_hops=2, node2vec_epochs=10),
]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in DATASETS:
        compare_csv = OUT_DIR / f"review_{cfg.name}_full9.csv"
        ablation_csv = OUT_DIR / f"review_{cfg.name}_ablation.csv"

        common = [
            "--dataset",
            cfg.name,
            "--data-root",
            "data",
            "--seeds",
            "3",
            "--feature-type",
            "node2vec",
            "--classification-loss",
            "bce",
            "--epochs",
            str(cfg.epochs),
            "--patience",
            str(cfg.patience),
            "--num-hops",
            str(cfg.num_hops),
            "--node2vec-epochs",
            str(cfg.node2vec_epochs),
            "--lambda-uncertainty",
            "0.08",
            "--lambda-ranking",
            "0.06",
            "--u-alpha",
            "0.6",
        ]

        run(
            [
                sys.executable,
                "main_link.py",
                *common,
                "--models",
                "ours,mc_dropout,guardian,trustguard,gatrust,gcn,sage,gat,mlp",
                "--out-csv",
                str(compare_csv),
                "--export-uncertainty-json",
                str(compare_csv.with_suffix(".json")),
            ]
        )
        run(
            [
                sys.executable,
                "main_link.py",
                *common,
                "--models",
                "ours,ours_nounc,ours_1hop,ours_notemp",
                "--out-csv",
                str(ablation_csv),
                "--export-uncertainty-json",
                str(ablation_csv.with_suffix(".json")),
            ]
        )

    run([sys.executable, "scripts/compile_reviewer_results.py"])
    run([sys.executable, "scripts/export_dataset_stats.py"])
    run([sys.executable, "scripts/generate_reviewer_prompts.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
