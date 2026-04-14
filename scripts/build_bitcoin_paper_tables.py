#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "paper_bitcoin"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: float) -> str:
    return f"{v:.4f}"


def replace_ours(base_rows: List[Dict[str, str]], ours_row: Dict[str, str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in base_rows:
        if r["model"] == "ours":
            nr = dict(r)
            for k, v in ours_row.items():
                if k in nr:
                    nr[k] = v
            out.append(nr)
        else:
            out.append(dict(r))
    return out


def best_value(rows: List[Dict[str, str]], key: str, higher_better: bool) -> float:
    vals = [float(r[key]) for r in rows]
    return max(vals) if higher_better else min(vals)


def build_summary(otc_rows: List[Dict[str, str]], alpha_rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("# Bitcoin Paper Summary (Tuned Ours)")
    lines.append("")
    for ds_name, rows in [("Bitcoin-OTC", otc_rows), ("Bitcoin-Alpha", alpha_rows)]:
        lines.append(f"## {ds_name}")
        ours = [r for r in rows if r["model"] == "ours"][0]
        lines.append(
            f"- Ours: AUC-PR {fmt(float(ours['auc_pr_mean']))}, "
            f"MCC {fmt(float(ours['mcc_mean']))}, "
            f"AURC(conf) {fmt(float(ours['aurc_confidence_mean']))}, "
            f"ECE {fmt(float(ours['ece_after_mean']))}"
        )

        auc_best = best_value(rows, "auc_pr_mean", higher_better=True)
        mcc_best = best_value(rows, "mcc_mean", higher_better=True)
        aurc_conf_best = best_value(rows, "aurc_confidence_mean", higher_better=False)
        ece_best = best_value(rows, "ece_after_mean", higher_better=False)
        lines.append(
            f"- Best values in table: AUC-PR {fmt(auc_best)}, MCC {fmt(mcc_best)}, "
            f"AURC(conf) {fmt(aurc_conf_best)}, ECE {fmt(ece_best)}"
        )

        tg = [r for r in rows if r["model"] == "trustguard"][0]
        ga = [r for r in rows if r["model"] == "gatrust"][0]
        lines.append(
            f"- vs TrustGuard-like: "
            f"+{fmt(float(ours['auc_pr_mean']) - float(tg['auc_pr_mean']))} AUC-PR, "
            f"+{fmt(float(ours['mcc_mean']) - float(tg['mcc_mean']))} MCC, "
            f"{fmt(float(tg['aurc_confidence_mean']) - float(ours['aurc_confidence_mean']))} lower AURC(conf)"
        )
        lines.append(
            f"- vs GAtrust-like: "
            f"+{fmt(float(ours['auc_pr_mean']) - float(ga['auc_pr_mean']))} AUC-PR, "
            f"+{fmt(float(ours['mcc_mean']) - float(ga['mcc_mean']))} MCC, "
            f"{fmt(float(ga['aurc_confidence_mean']) - float(ours['aurc_confidence_mean']))} lower AURC(conf)"
        )
        lines.append("")

    lines.append("## Claim Boundary")
    lines.append("- Strong: Ours shows robust gains over trust-specific baselines (TrustGuard-like, GAtrust-like) on AUC-PR and MCC in both Bitcoin datasets.")
    lines.append("- Moderate: Ours is top or near-top on AUC-PR, but not uniformly best on every uncertainty/calibration metric against all generic GNN baselines.")
    lines.append("- Avoid: claiming universal best performance across all metrics.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    base_otc = read_csv(REPO_ROOT / "outputs" / "review_otc_full9_agg.csv")
    base_alpha = read_csv(REPO_ROOT / "outputs" / "review_alpha_full9_agg.csv")
    ours_otc = read_csv(REPO_ROOT / "outputs" / "ours_otc_tuned_s3_agg.csv")[0]
    ours_alpha = read_csv(REPO_ROOT / "outputs" / "ours_alpha_tuned_s3_agg.csv")[0]

    merged_otc = replace_ours(base_otc, ours_otc)
    merged_alpha = replace_ours(base_alpha, ours_alpha)
    for r in merged_otc:
        r["dataset"] = "bitcoin_otc"
    for r in merged_alpha:
        r["dataset"] = "bitcoin_alpha"
    merged_all = merged_otc + merged_alpha
    write_csv(OUT_DIR / "bitcoin_compare_tuned_agg.csv", merged_all)

    ab_otc = read_csv(REPO_ROOT / "outputs" / "ours_otc_tuned_ablation_s3_agg.csv")
    ab_alpha = read_csv(REPO_ROOT / "outputs" / "ours_alpha_tuned_ablation_s3_agg.csv")
    for r in ab_otc:
        r["dataset"] = "bitcoin_otc"
    for r in ab_alpha:
        r["dataset"] = "bitcoin_alpha"
    write_csv(OUT_DIR / "bitcoin_ablation_tuned_agg.csv", ab_otc + ab_alpha)

    summary_md = build_summary(merged_otc, merged_alpha)
    (OUT_DIR / "paper_summary.md").write_text(summary_md, encoding="utf-8")
    print(f"Saved: {OUT_DIR / 'bitcoin_compare_tuned_agg.csv'}")
    print(f"Saved: {OUT_DIR / 'bitcoin_ablation_tuned_agg.csv'}")
    print(f"Saved: {OUT_DIR / 'paper_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
