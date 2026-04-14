#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "reviewer_pack"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.temporal_split import build_temporal_dataset


def dataset_stats_row(dataset: str) -> dict:
    split = build_temporal_dataset(dataset=dataset, data_root=str(REPO_ROOT / "data"))
    total_edges = int(split.train_edge_label.numel() + split.val_edge_label.numel() + split.test_edge_label.numel())
    pos_total = int((split.train_edge_label == 1).sum().item() + (split.val_edge_label == 1).sum().item() + (split.test_edge_label == 1).sum().item())
    neg_total = total_edges - pos_total
    return {
        "dataset": dataset,
        "num_nodes": str(int(split.num_nodes)),
        "num_edges_total": str(total_edges),
        "num_edges_train": str(int(split.train_edge_label.numel())),
        "num_edges_val": str(int(split.val_edge_label.numel())),
        "num_edges_test": str(int(split.test_edge_label.numel())),
        "positive_ratio_total": f"{(pos_total / max(total_edges, 1)):.6f}",
        "negative_ratio_total": f"{(neg_total / max(total_edges, 1)):.6f}",
        "train_end_ts": str(int(split.train_end_ts)),
        "val_end_ts": str(int(split.val_end_ts)),
    }


def main() -> int:
    rows = [dataset_stats_row(ds) for ds in ("bitcoin_otc", "bitcoin_alpha", "wiki_rfa")]
    out_path = OUT_DIR / "dataset_stats.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved dataset stats: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
