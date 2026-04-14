#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
REVIEW_DIR = OUT_DIR / "reviewer_pack"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)


COMPARE_FILES = {
    "bitcoin_otc": OUT_DIR / "review_otc_full9.csv",
    "bitcoin_alpha": OUT_DIR / "review_alpha_full9.csv",
    "wiki_rfa": OUT_DIR / "review_wikirfa_full9.csv",
}
ABLATION_FILES = {
    "bitcoin_otc": OUT_DIR / "review_otc_ablation.csv",
    "bitcoin_alpha": OUT_DIR / "review_alpha_ablation.csv",
    "wiki_rfa": OUT_DIR / "review_wikirfa_ablation.csv",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / len(values)
    return m, math.sqrt(v)


def aggregate(rows: List[Dict[str, str]], metrics: List[str]) -> Dict[Tuple[str, str], Dict[str, float]]:
    groups: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for r in rows:
        key = (r["dataset"], r["model"])
        groups.setdefault(key, {m: [] for m in metrics})
        for m in metrics:
            groups[key][m].append(float(r[m]))
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, vals in groups.items():
        out[key] = {}
        for m in metrics:
            mm, ss = mean_std(vals[m])
            out[key][f"{m}_mean"] = mm
            out[key][f"{m}_std"] = ss
    return out


def exact_sign_flip_pvalue(diffs: List[float]) -> float:
    """
    Two-sided exact randomization test by sign-flip enumeration.
    """
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = abs(sum(diffs))
    total = 0
    ge = 0
    for bits in itertools.product([-1.0, 1.0], repeat=n):
        s = abs(sum(d * b for d, b in zip(diffs, bits)))
        total += 1
        if s >= obs - 1e-12:
            ge += 1
    return ge / total


def paired_test_rows(
    rows: List[Dict[str, str]],
    dataset: str,
    baseline: str,
    metric: str,
    higher_better: bool,
) -> Dict[str, str]:
    by_seed: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r["dataset"] != dataset:
            continue
        if r["model"] not in ("ours", baseline):
            continue
        by_seed.setdefault(r["seed"], {})
        by_seed[r["seed"]][r["model"]] = float(r[metric])

    diffs: List[float] = []
    for _seed, rec in sorted(by_seed.items()):
        if "ours" in rec and baseline in rec:
            d = rec["ours"] - rec[baseline] if higher_better else rec[baseline] - rec["ours"]
            diffs.append(d)

    if not diffs:
        return {
            "dataset": dataset,
            "baseline": baseline,
            "metric": metric,
            "mean_diff_good_direction": "nan",
            "p_value_exact_sign_flip": "nan",
            "n_pairs": "0",
            "claim_strength": "insufficient_pairs",
        }

    m, s = mean_std(diffs)
    p = exact_sign_flip_pvalue(diffs)
    if m > 0 and p < 0.05:
        strength = "strong"
    elif m > 0:
        strength = "weak"
    else:
        strength = "negative"
    return {
        "dataset": dataset,
        "baseline": baseline,
        "metric": metric,
        "mean_diff_good_direction": f"{m:.6f}",
        "std_diff": f"{s:.6f}",
        "p_value_exact_sign_flip": f"{p:.6f}",
        "n_pairs": str(len(diffs)),
        "claim_strength": strength,
    }


def paired_test_rows_all(
    rows: List[Dict[str, str]],
    baseline: str,
    metric: str,
    higher_better: bool,
) -> Dict[str, str]:
    by_key: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        if r["model"] not in ("ours", baseline):
            continue
        key = (r["dataset"], r["seed"])
        by_key.setdefault(key, {})
        by_key[key][r["model"]] = float(r[metric])

    diffs: List[float] = []
    for _key, rec in sorted(by_key.items()):
        if "ours" in rec and baseline in rec:
            d = rec["ours"] - rec[baseline] if higher_better else rec[baseline] - rec["ours"]
            diffs.append(d)

    if not diffs:
        return {
            "dataset": "all_datasets",
            "baseline": baseline,
            "metric": metric,
            "mean_diff_good_direction": "nan",
            "p_value_exact_sign_flip": "nan",
            "n_pairs": "0",
            "claim_strength": "insufficient_pairs",
        }

    m, s = mean_std(diffs)
    p = exact_sign_flip_pvalue(diffs)
    if m > 0 and p < 0.05:
        strength = "strong"
    elif m > 0:
        strength = "weak"
    else:
        strength = "negative"
    return {
        "dataset": "all_datasets",
        "baseline": baseline,
        "metric": metric,
        "mean_diff_good_direction": f"{m:.6f}",
        "std_diff": f"{s:.6f}",
        "p_value_exact_sign_flip": f"{p:.6f}",
        "n_pairs": str(len(diffs)),
        "claim_strength": strength,
    }


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(
    compare_agg_rows: List[Dict[str, str]],
    ablation_agg_rows: List[Dict[str, str]],
    sig_rows: List[Dict[str, str]],
    path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Reviewer-Aligned Revision Summary")
    lines.append("")
    lines.append("This pack reports three-dataset results under one protocol and distinguishes strong vs weak claims.")
    lines.append("")
    lines.append("## Main Comparison (mean±std)")
    lines.append("")
    for r in compare_agg_rows:
        lines.append(
            f"- {r['dataset']} | {r['model']}: "
            f"AUC-PR {r['auc_pr_mean']}+/-{r['auc_pr_std']}, "
            f"MCC {r['mcc_mean']}+/-{r['mcc_std']}, "
            f"AURC(primary) {r['aurc_mean']}+/-{r['aurc_std']}, "
            f"AURC(confidence) {r['aurc_confidence_mean']}+/-{r['aurc_confidence_std']}, "
            f"ECE {r['ece_after_mean']}+/-{r['ece_after_std']}"
        )
    lines.append("")
    lines.append("## Ablation (ours-family)")
    lines.append("")
    for r in ablation_agg_rows:
        lines.append(
            f"- {r['dataset']} | {r['model']}: "
            f"AUC-PR {r['auc_pr_mean']}+/-{r['auc_pr_std']}, "
            f"MCC {r['mcc_mean']}+/-{r['mcc_std']}, "
            f"AURC(primary) {r['aurc_mean']}+/-{r['aurc_std']}"
        )
    lines.append("")
    lines.append("## Claim Strength (paired exact sign-flip)")
    lines.append("")
    for r in sig_rows:
        lines.append(
            f"- {r['dataset']} | ours vs {r['baseline']} | {r['metric']}: "
            f"diff={r['mean_diff_good_direction']}, p={r['p_value_exact_sign_flip']}, "
            f"n={r['n_pairs']}, strength={r['claim_strength']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    compare_rows: List[Dict[str, str]] = []
    for ds, fp in COMPARE_FILES.items():
        if not fp.is_file():
            continue
        rows = read_csv(fp)
        for r in rows:
            r["dataset"] = ds
        compare_rows.extend(rows)

    ablation_rows: List[Dict[str, str]] = []
    for ds, fp in ABLATION_FILES.items():
        if not fp.is_file():
            continue
        rows = read_csv(fp)
        for r in rows:
            r["dataset"] = ds
        ablation_rows.extend(rows)

    if not compare_rows:
        print("No compare CSV found.")
        return 2

    compare_metrics = [
        "auc_pr",
        "mcc",
        "aurc",
        "aurc_confidence",
        "ece_after",
        "brier",
    ]
    ablation_metrics = ["auc_pr", "mcc", "aurc", "ece_after"]

    compare_agg = aggregate(compare_rows, compare_metrics)
    ablation_agg = aggregate(ablation_rows, ablation_metrics) if ablation_rows else {}

    compare_agg_rows: List[Dict[str, str]] = []
    for (dataset, model), vals in sorted(compare_agg.items()):
        row = {"dataset": dataset, "model": model}
        for k, v in vals.items():
            row[k] = f"{v:.6f}"
        compare_agg_rows.append(row)

    ablation_agg_rows: List[Dict[str, str]] = []
    for (dataset, model), vals in sorted(ablation_agg.items()):
        row = {"dataset": dataset, "model": model}
        for k, v in vals.items():
            row[k] = f"{v:.6f}"
        ablation_agg_rows.append(row)

    write_csv(REVIEW_DIR / "main_compare_multidataset_agg.csv", compare_agg_rows)
    if ablation_agg_rows:
        write_csv(REVIEW_DIR / "ablation_multidataset_agg.csv", ablation_agg_rows)

    sig_rows: List[Dict[str, str]] = []
    baselines = ["mc_dropout", "trustguard", "gatrust", "guardian"]
    metrics = [("auc_pr", True), ("mcc", True), ("aurc", False), ("aurc_confidence", False)]
    datasets = sorted(set(r["dataset"] for r in compare_rows))
    for ds in datasets:
        for bl in baselines:
            for metric, higher_better in metrics:
                sig_rows.append(paired_test_rows(compare_rows, ds, bl, metric, higher_better))
    for bl in baselines:
        for metric, higher_better in metrics:
            sig_rows.append(paired_test_rows_all(compare_rows, bl, metric, higher_better))

    write_csv(REVIEW_DIR / "paired_significance.csv", sig_rows)
    write_markdown_summary(
        compare_agg_rows=compare_agg_rows,
        ablation_agg_rows=ablation_agg_rows,
        sig_rows=sig_rows,
        path=REVIEW_DIR / "reviewer_summary.md",
    )
    print(f"Saved reviewer pack to: {REVIEW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
