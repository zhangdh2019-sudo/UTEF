from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_assets_full"
OUT.mkdir(parents=True, exist_ok=True)


def fmt(mean: float, std: float, d: int = 4) -> str:
    return f"{mean:.{d}f} ± {std:.{d}f}"


def load():
    compare_agg = pd.read_csv(ROOT / "outputs" / "full9_compare_agg.csv")
    ablation_agg = pd.read_csv(ROOT / "outputs" / "full_ablation_agg.csv")
    compare_json = json.loads((ROOT / "outputs" / "full9_compare.json").read_text(encoding="utf-8"))
    ablation_json = json.loads((ROOT / "outputs" / "full_ablation.json").read_text(encoding="utf-8"))
    return compare_agg, ablation_agg, compare_json, ablation_json


def model_order_compare(df: pd.DataFrame) -> list[str]:
    desired = ["node2vec_mlp", "gcn", "sage", "gat", "gatrust", "trustguard", "guardian", "mc_dropout", "ours"]
    avail = set(df["model"].tolist())
    return [m for m in desired if m in avail]


def model_order_ablation(df: pd.DataFrame) -> list[str]:
    desired = ["ours", "ours_nounc", "ours_1hop", "ours_notemp"]
    avail = set(df["model"].tolist())
    return [m for m in desired if m in avail]


def build_rc_mean(rc_json: dict, desired_models: list[str], out_name: str) -> pd.DataFrame:
    rows = []
    for m in desired_models:
        curves = []
        cover = None
        for seed_blob in rc_json["seeds"]:
            if m not in seed_blob["models"]:
                continue
            rc = seed_blob["models"][m]["risk_coverage"]
            c = np.asarray(rc["coverage"], dtype=float)
            r = np.asarray(rc["risk"], dtype=float)
            cover = c if cover is None else cover
            curves.append(r)
        if not curves:
            continue
        stack = np.stack(curves, axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        for c, mu, sd in zip(cover, mean, std):
            rows.append({"model": m, "coverage": float(c), "risk_mean": float(mu), "risk_std": float(sd)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / out_name, index=False, encoding="utf-8-sig")
    return out_df


def write_tables(compare_agg: pd.DataFrame, ablation_agg: pd.DataFrame) -> None:
    c_order = model_order_compare(compare_agg)
    a_order = model_order_ablation(ablation_agg)

    cdf = compare_agg.set_index("model").loc[c_order].reset_index()
    adf = ablation_agg.set_index("model").loc[a_order].reset_index()

    table_compare = pd.DataFrame(
        {
            "Model": cdf["model"],
            "AUC-PR": [fmt(m, s) for m, s in zip(cdf["auc_pr_mean"], cdf["auc_pr_std"])],
            "MCC": [fmt(m, s) for m, s in zip(cdf["mcc_mean"], cdf["mcc_std"])],
            "ECE": [fmt(m, s) for m, s in zip(cdf["ece_after_mean"], cdf["ece_after_std"])],
            "AURC": [fmt(m, s) for m, s in zip(cdf["aurc_mean"], cdf["aurc_std"])],
            "RC Spearman": [fmt(m, s) for m, s in zip(cdf["rc_spearman_cov_risk_mean"], cdf["rc_spearman_cov_risk_std"])],
            "Filt20 Risk": [fmt(m, s) for m, s in zip(cdf["filt20_risk_mean"], cdf["filt20_risk_std"])],
        }
    )
    table_compare.to_csv(OUT / "full_compare_table.csv", index=False, encoding="utf-8-sig")
    (OUT / "full_compare_table.md").write_text(table_compare.to_markdown(index=False), encoding="utf-8")

    table_ablation = pd.DataFrame(
        {
            "Variant": adf["model"],
            "AUC-PR": [fmt(m, s) for m, s in zip(adf["auc_pr_mean"], adf["auc_pr_std"])],
            "MCC": [fmt(m, s) for m, s in zip(adf["mcc_mean"], adf["mcc_std"])],
            "ECE": [fmt(m, s) for m, s in zip(adf["ece_after_mean"], adf["ece_after_std"])],
            "AURC": [fmt(m, s) for m, s in zip(adf["aurc_mean"], adf["aurc_std"])],
            "RC Spearman": [fmt(m, s) for m, s in zip(adf["rc_spearman_cov_risk_mean"], adf["rc_spearman_cov_risk_std"])],
        }
    )
    table_ablation.to_csv(OUT / "full_ablation_table.csv", index=False, encoding="utf-8-sig")
    (OUT / "full_ablation_table.md").write_text(table_ablation.to_markdown(index=False), encoding="utf-8")


def _row_prompt(df: pd.DataFrame, model: str) -> str:
    r = df[df["model"] == model].iloc[0]
    return (
        f"- {model}: AUC-PR {r['auc_pr_mean']:.4f} ± {r['auc_pr_std']:.4f}, "
        f"MCC {r['mcc_mean']:.4f} ± {r['mcc_std']:.4f}, "
        f"ECE {r['ece_after_mean']:.4f} ± {r['ece_after_std']:.4f}, "
        f"AURC {r['aurc_mean']:.4f} ± {r['aurc_std']:.4f}, "
        f"RC Spearman {r['rc_spearman_cov_risk_mean']:.4f} ± {r['rc_spearman_cov_risk_std']:.4f}, "
        f"Filt20 Risk {r['filt20_risk_mean']:.4f} ± {r['filt20_risk_std']:.4f}"
    )


def write_prompts(compare_agg: pd.DataFrame, ablation_agg: pd.DataFrame) -> None:
    c_order = model_order_compare(compare_agg)
    a_order = model_order_ablation(ablation_agg)

    compare_lines = "\n".join(_row_prompt(compare_agg, m) for m in c_order)
    ablation_lines = "\n".join(_row_prompt(ablation_agg, m) for m in a_order)

    text = f"""# Full Experiment Prompts (Real run, unified protocol)

## Prompt 1: Full comparison figure (9 models)
Create an IEEE-style publication figure for temporal trust prediction main comparison.

Methods (exact order):
- Node2Vec+MLP (`node2vec_mlp`)
- GCN
- GraphSAGE (`sage`)
- GAT
- GATrust-like (`gatrust`)
- TrustGuard-like (`trustguard`)
- Guardian (`guardian`)
- MC Dropout (`mc_dropout`)
- Ours (`ours`)

Use mean ± std error bars from this real run:
{compare_lines}

Subplots:
1. AUC-PR (higher is better)
2. MCC (higher is better)
3. ECE (lower is better)
4. AURC (lower is better)
5. RC Spearman (higher is better)

Requirements:
- journal style, white background, muted colors
- keep Ours visually identifiable but do not falsify ranking
- do not alter numeric values

## Prompt 2: Full ablation figure
Create an IEEE-style ablation figure using these variants:
- Full (`ours`)
- w/o uncertainty (`ours_nounc`)
- w/o multi-hop (`ours_1hop`)
- w/o temporal (`ours_notemp`)

Use mean ± std error bars from this real run:
{ablation_lines}

Subplots:
1. AUC-PR
2. MCC
3. ECE
4. AURC
5. RC Spearman

Requirements:
- keep values exact
- emphasize that core predictive metrics are AUC-PR and MCC

## Prompt 3: Full risk-coverage figure (9 models)
Create an IEEE-style risk-coverage plot using per-coverage mean ± std risk values from:
`outputs/paper_assets_full/full_compare_rc_mean.csv`

Methods:
- node2vec_mlp, gcn, sage, gat, gatrust, trustguard, guardian, mc_dropout, ours

Plot:
- x-axis: Coverage
- y-axis: Risk
- one curve per method + shaded std band
- use exact data from the CSV file; no smoothing that changes values

## Prompt 4: Ablation risk-coverage figure
Create an IEEE-style risk-coverage plot using:
`outputs/paper_assets_full/full_ablation_rc_mean.csv`

Methods:
- ours, ours_nounc, ours_1hop, ours_notemp

Same requirements as Prompt 3.

## Prompt 5: Main comparison table
Create an IEEE-style table from:
`outputs/paper_assets_full/full_compare_table.csv`

## Prompt 6: Ablation table
Create an IEEE-style table from:
`outputs/paper_assets_full/full_ablation_table.csv`
"""
    (OUT / "full_prompts.md").write_text(text, encoding="utf-8")


def main():
    compare_agg, ablation_agg, compare_json, ablation_json = load()
    write_tables(compare_agg, ablation_agg)
    build_rc_mean(compare_json, model_order_compare(compare_agg), "full_compare_rc_mean.csv")
    build_rc_mean(ablation_json, model_order_ablation(ablation_agg), "full_ablation_rc_mean.csv")
    write_prompts(compare_agg, ablation_agg)
    print(f"written to: {OUT}")


if __name__ == "__main__":
    main()
