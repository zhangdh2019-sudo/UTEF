#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK = REPO_ROOT / "outputs" / "reviewer_pack"


def main() -> int:
    out = PACK / "figure_table_prompts.md"
    lines = []
    lines.append("# Figure/Table Prompts (Reviewer-Aligned, Real Data)")
    lines.append("")
    lines.append("All prompts below must use only these result files:")
    lines.append(f"- `{(PACK / 'dataset_stats.csv').as_posix()}`")
    lines.append(f"- `{(PACK / 'main_compare_multidataset_agg.csv').as_posix()}`")
    lines.append(f"- `{(PACK / 'ablation_multidataset_agg.csv').as_posix()}`")
    lines.append(f"- `{(PACK / 'paired_significance.csv').as_posix()}`")
    lines.append(f"- `{(REPO_ROOT / 'outputs' / 'review_otc_full9.json').as_posix()}`")
    lines.append(f"- `{(REPO_ROOT / 'outputs' / 'review_alpha_full9.json').as_posix()}`")
    lines.append(f"- `{(REPO_ROOT / 'outputs' / 'review_wikirfa_full9.json').as_posix()}`")
    lines.append("")
    lines.append("## Table 1: Dataset Statistics")
    lines.append("Prompt:")
    lines.append("Create an IEEE-style table named 'Dataset Statistics'. Use only dataset_stats.csv. Columns: Dataset, #Nodes, #Edges, Train/Val/Test edges, Positive ratio, Negative ratio. Keep 3 decimal places for ratios.")
    lines.append("")
    lines.append("## Table 2: Main Comparison Across 3 Datasets")
    lines.append("Prompt:")
    lines.append("Create a multi-dataset comparison table from main_compare_multidataset_agg.csv. Rows are methods, grouped by dataset (Bitcoin-OTC, Bitcoin-Alpha, Wiki-RfA). Report mean+/-std for AUC-PR, MCC, AURC(primary), AURC(confidence), ECE. Bold the best value per metric per dataset (AUC-PR/MCC higher is better; AURC/ECE lower is better).")
    lines.append("")
    lines.append("## Table 3: Ablation Study")
    lines.append("Prompt:")
    lines.append("Create an ablation table from ablation_multidataset_agg.csv. For each dataset, compare Ours, Ours-NoUnc, Ours-1Hop, Ours-NoTemp on AUC-PR, MCC, AURC(primary), ECE. Highlight whether removing each module degrades AUC-PR or selective risk.")
    lines.append("")
    lines.append("## Table 4: Paired Significance")
    lines.append("Prompt:")
    lines.append("Create a statistical significance table from paired_significance.csv. Keep only ours vs {MC Dropout, TrustGuard-like, GAtrust-like}. Columns: Dataset, Metric, Mean diff (good direction), p-value (exact sign-flip), Claim strength. Add a note that n=3 seeds per dataset and significance is limited.")
    lines.append("")
    lines.append("## Figure 1: System Model")
    lines.append("Prompt:")
    lines.append("Draw a publication-quality system diagram for a temporal signed trust evaluation model. Pipeline blocks: (1) Temporal signed graph split (train-only propagation graph), (2) Signed multi-hop encoder (A+ and A- channels), (3) Edge temporal context encoder (delta_u, delta_v, pos/neg counts, last pair interval), (4) Edge evidential decoder (Dirichlet evidence), (5) Global hop evidence fusion, (6) Probability + uncertainty outputs, (7) Dual decision protocol: uncertainty-based selective prediction and confidence-based selective prediction. Use a clean IEEE style, white background, blue-gray palette, vector-like sharp edges, and concise math labels.")
    lines.append("")
    lines.append("## Figure 2-4: Risk-Coverage Curves (Per Dataset)")
    lines.append("Prompt template:")
    lines.append("Using review_<dataset>_full9.json, plot mean risk-coverage curves across seeds for Ours, TrustGuard-like, MC Dropout under (a) primary uncertainty ranking and (b) confidence ranking. X-axis: coverage, Y-axis: risk=1-accuracy. Include AURC in legend. Use two subplots per dataset.")
    lines.append("")
    lines.append("## Figure 5: Confidence vs Uncertainty AURC Gap")
    lines.append("Prompt:")
    lines.append("From main_compare_multidataset_agg.csv, create a grouped bar chart of AURC(primary) and AURC(confidence) for Ours, TrustGuard-like, MC Dropout on each dataset. Lower is better. Annotate delta(AURC_primary - AURC_confidence).")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved prompts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
