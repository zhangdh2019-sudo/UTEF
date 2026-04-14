#!/usr/bin/env python3
"""
Logical consistency checks on main_link.py raw CSV rows.

Usage:
  python scripts/validate_experiment_metrics.py outputs/paper_run_5seed.csv
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/link_results.csv")
    if not path.is_file():
        print(f"Missing file: {path}")
        return 2

    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    errors: List[str] = []
    warns: List[str] = []

    for i, r in enumerate(rows, start=2):
        def cell(k: str) -> float:
            return float(r[k])

        # Core bounds
        for k in ("auc_pr", "auc_roc", "balanced_accuracy", "specificity", "pred_pos_rate"):
            v = cell(k)
            if not (0.0 <= v <= 1.0):
                errors.append(f"row {i} {k}={v} not in [0,1]")

        # MCC in [-1,1]
        mcc = cell("mcc")
        if not (-1.01 <= mcc <= 1.01):
            errors.append(f"row {i} mcc={mcc} out of [-1,1]")

        # Risk + accuracy = 1 (filtering)
        for pref in ("filt10", "filt20", "filt30"):
            acc = cell(f"{pref}_acc")
            risk = cell(f"{pref}_risk")
            if _finite(acc) and _finite(risk):
                s = acc + risk
                if abs(s - 1.0) > 1e-3:
                    errors.append(f"row {i} {pref}_acc+{pref}_risk={s} (expect 1.0)")

        # ECE in [0,1]
        for k in ("ece_before", "ece_after"):
            e = cell(k)
            if not (0.0 <= e <= 1.0):
                errors.append(f"row {i} {k}={e} not in [0,1]")

        # Brier in [0,1] for binary
        b = cell("brier")
        if not (0.0 <= b <= 1.0):
            errors.append(f"row {i} brier={b} not in [0,1]")

        # TS usually helps ECE on expectation (not strict on test)
        if cell("ece_after") > cell("ece_before") + 0.08:
            warns.append(f"row {i} ece_after >> ece_before (TS may not have helped much on test)")

    # Per-model aggregates for 5-seed stability hints
    by_model: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(float(r["ece_after"]))

    print(f"Validated {len(rows)} rows from {path}\n")
    if errors:
        print("ERRORS (fix code or protocol):")
        for e in errors[:50]:
            print("  -", e)
        if len(errors) > 50:
            print(f"  ... ({len(errors)} total)")
        print()
    else:
        print("OK: no hard constraint violations.\n")

    if warns:
        print("WARNINGS:")
        for w in warns[:20]:
            print("  -", w)
        print()

    # Optional: compare AURC ours vs baselines on mean (need agg file)
    agg_path = path.with_name(path.stem + "_agg.csv")
    if agg_path.is_file():
        print(f"--- Quick read: {agg_path.name} ---")
        with open(agg_path, newline="", encoding="utf-8") as f:
            agg = list(csv.DictReader(f))
        aurc = {a["model"]: float(a["aurc_mean"]) for a in agg if a.get("aurc_mean")}
        if "ours" in aurc:
            bl = [aurc[m] for m in ("gatrust", "trustguard", "evidential_mlp") if m in aurc]
            if bl:
                best_bl = min(bl)
                print(f"AURC ours mean={aurc['ours']:.4f}  min(graph+evid baselines)={best_bl:.4f}")
                if aurc["ours"] <= best_bl:
                    print("  => ours AURC <= min baseline (strong selective headline).")
                else:
                    gap = aurc["ours"] - best_bl
                    print(f"  => ours AURC is {gap:+.4f} vs best baseline; emphasize ECE + filtering lift + temporal graph.")
            if "trustguard" in aurc:
                print(
                    f"  (vs TrustGuard-style only: ours {aurc['ours']:.4f} vs trustguard {aurc['trustguard']:.4f})"
                )

        # Filtering lift: filt20 accuracy vs full-test balanced accuracy (ours)
        with open(path, newline="", encoding="utf-8") as f2:
            rows = list(csv.DictReader(f2))
        ours_rows = [x for x in rows if x.get("model") == "ours"]
        if len(ours_rows) >= 2:
            import statistics as stats

            ba = [float(x["balanced_accuracy"]) for x in ours_rows]
            fa = [float(x["filt20_acc"]) for x in ours_rows]
            lift = stats.mean(fa) - stats.mean(ba)
            print(f"\nOurs filtering (remove top 20%% unc): mean acc lift vs full balanced_acc = {lift*100:.2f} pp")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
