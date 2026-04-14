import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.temporal_split import build_temporal_bitcoin_otc


OUT = ROOT / "outputs" / "paper_assets"
OUT.mkdir(parents=True, exist_ok=True)


def _fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _save_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"{stem}.md").write_text(df.to_markdown(index=False), encoding="utf-8")


def load_data():
    compare_agg = pd.read_csv(ROOT / "outputs" / "paper_best_compare_agg.csv")
    ablation_agg = pd.read_csv(ROOT / "outputs" / "paper_best_ablation_agg.csv")
    compare_raw = pd.read_csv(ROOT / "outputs" / "paper_best_compare.csv")
    repro_json = json.loads((ROOT / "outputs" / "paper_assets_repro.json").read_text(encoding="utf-8"))
    return compare_agg, ablation_agg, compare_raw, repro_json


def build_tables(compare_agg: pd.DataFrame, ablation_agg: pd.DataFrame) -> None:
    split = build_temporal_bitcoin_otc(root=str(ROOT / "data"))
    train_y = split.train_edge_label.cpu().numpy()
    val_y = split.val_edge_label.cpu().numpy()
    test_y = split.test_edge_label.cpu().numpy()

    table1 = pd.DataFrame(
        [
            {
                "Dataset": "Bitcoin-OTC",
                "#Nodes": int(split.num_nodes),
                "#Train Edges": int(split.train_edge_index.size(1)),
                "#Val Edges": int(split.val_edge_index.size(1)),
                "#Test Edges": int(split.test_edge_index.size(1)),
                "Train Pos/Neg": f"{int((train_y == 1).sum())}/{int((train_y == 0).sum())}",
                "Val Pos/Neg": f"{int((val_y == 1).sum())}/{int((val_y == 0).sum())}",
                "Test Pos/Neg": f"{int((test_y == 1).sum())}/{int((test_y == 0).sum())}",
            }
        ]
    )
    _save_table(table1, "table1_dataset_statistics")

    main_rows = []
    for _, r in compare_agg.sort_values("model").iterrows():
        main_rows.append(
            {
                "Model": r["model"],
                "AUC-PR": _fmt_pm(r["auc_pr_mean"], r["auc_pr_std"]),
                "MCC": _fmt_pm(r["mcc_mean"], r["mcc_std"]),
                "ECE": _fmt_pm(r["ece_after_mean"], r["ece_after_std"]),
                "AURC": _fmt_pm(r["aurc_mean"], r["aurc_std"]),
                "SelAcc@80": _fmt_pm(r["sel_acc_80_mean"], r["sel_acc_80_std"]),
                "Filt20 Risk": _fmt_pm(r["filt20_risk_mean"], r["filt20_risk_std"]),
                "RC Spearman": _fmt_pm(r["rc_spearman_cov_risk_mean"], r["rc_spearman_cov_risk_std"]),
                "Pred Pos Rate": _fmt_pm(r["pred_pos_rate_mean"], r["pred_pos_rate_std"]),
            }
        )
    _save_table(pd.DataFrame(main_rows), "table2_main_comparison")

    ab_rows = []
    order = ["ours", "ours_nounc", "ours_1hop", "ours_notemp"]
    ablation_agg = ablation_agg.set_index("model").loc[order].reset_index()
    for _, r in ablation_agg.iterrows():
        ab_rows.append(
            {
                "Variant": r["model"],
                "AUC-PR": _fmt_pm(r["auc_pr_mean"], r["auc_pr_std"]),
                "MCC": _fmt_pm(r["mcc_mean"], r["mcc_std"]),
                "ECE": _fmt_pm(r["ece_after_mean"], r["ece_after_std"]),
                "AURC": _fmt_pm(r["aurc_mean"], r["aurc_std"]),
                "RC Spearman": _fmt_pm(r["rc_spearman_cov_risk_mean"], r["rc_spearman_cov_risk_std"]),
            }
        )
    _save_table(pd.DataFrame(ab_rows), "table3_ablation")


def plot_system_model() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(16.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "blue": "#DCEBFA",
        "green": "#E2F3E8",
        "orange": "#FCE8D5",
        "purple": "#EFE4FA",
        "gray": "#F4F5F7",
        "line": "#2F4858",
    }

    def box(x, y, w, h, title, body, fc, ec="#314351", radius=0.02, title_size=14, body_size=11):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            linewidth=1.8,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.06, title, fontsize=title_size, fontweight="bold", va="top", color="#16212B")
        ax.text(x + 0.02, y + h - 0.12, body, fontsize=body_size, va="top", color="#22313F", linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=2.0, color=colors["line"], shrinkA=4, shrinkB=4),
        )

    ax.text(0.03, 0.95, "Figure 1. System model of the proposed uncertainty-aware temporal trust prediction framework", fontsize=18, fontweight="bold", va="top")

    box(
        0.04,
        0.60,
        0.22,
        0.23,
        "Temporal signed prefix graph",
        "Historical interactions only\n"
        "Positive edges: $\\widetilde{A}^{+}$\n"
        "Negative edges: $\\widetilde{A}^{-}$\n"
        "Future edges are never injected",
        colors["blue"],
    )
    box(
        0.31,
        0.57,
        0.23,
        0.28,
        "Signed multi-hop encoder",
        "$H^{(0)}=\\phi(XW_{in}+b)$\n"
        "$M^{+,k}=\\widetilde{A}^{+}H^{(k-1)}$\n"
        "$M^{-,k}=\\widetilde{A}^{-}H^{(k-1)}$\n"
        "$H^{(k)}=\\phi([M^{+,k}\\|M^{-,k}]W_f)$",
        colors["green"],
    )
    box(
        0.31,
        0.20,
        0.23,
        0.22,
        "Temporal edge context",
        "$g_{uv}(t)=[\\delta_u,\\delta_v,c_{uv}^{+},c_{uv}^{-},\\delta_{uv}]$\n"
        "Node recency, pair counts,\n"
        "and latest interaction gap",
        colors["gray"],
    )
    box(
        0.59,
        0.57,
        0.17,
        0.28,
        "Evidential decoder",
        "$z_{uv}^{(k)}=[h_u^{(k)}\\|h_v^{(k)}\\||h_u^{(k)}-h_v^{(k)}|\\|$\n"
        "$(h_u^{(k)}\\odot h_v^{(k)})\\|s_{uv}^{(k)}\\|g_{uv}(t)]$\n"
        "$e_{uv}^{(k)}=\\mathrm{Softplus}(\\mathrm{MLP}(z_{uv}^{(k)}))$",
        colors["orange"],
        body_size=10.3,
    )
    box(
        0.80,
        0.57,
        0.15,
        0.28,
        "Global hop fusion",
        "$\\pi_k=\\dfrac{e^{\\beta_k}}{\\sum_j e^{\\beta_j}}$\n"
        "$e_{uv}=\\sum_k \\pi_k e_{uv}^{(k)}$\n"
        "$\\alpha_{uv}=e_{uv}+\\mathbf{1}$\n"
        "shared hop weights",
        colors["purple"],
        body_size=10.5,
    )
    box(
        0.60,
        0.20,
        0.37,
        0.22,
        "Probability and uncertainty output",
        "$p_{uv,c}=\\alpha_{uv,c}/\\sum_j\\alpha_{uv,j}$\n"
        "$\\mathcal{U}_{uv}=\\lambda\\,\\mathcal{U}^{vac}_{uv}+(1-\\lambda)\\,\\mathcal{H}_{uv}$\n"
        "Output: trust probability + decision-aware uncertainty",
        colors["blue"],
        body_size=11,
    )

    arrow(0.26, 0.71, 0.31, 0.71)
    arrow(0.54, 0.71, 0.59, 0.71)
    arrow(0.76, 0.71, 0.80, 0.71)
    arrow(0.88, 0.57, 0.84, 0.42)
    arrow(0.54, 0.31, 0.60, 0.31)
    arrow(0.69, 0.57, 0.69, 0.42)

    ax.text(0.045, 0.52, "Strict temporal protocol", fontsize=11, fontweight="bold", color="#3A4C5A")
    ax.text(0.045, 0.48, "train prefix  →  val  →  test", fontsize=11, color="#3A4C5A")
    ax.text(0.60, 0.12, "Uncertainty is later used for selective prediction and risk-aware filtering.", fontsize=11, color="#3A4C5A")

    fig.tight_layout()
    fig.savefig(OUT / "figure1_system_model.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure1_system_model.svg", bbox_inches="tight")
    plt.close(fig)


def plot_main_comparison(compare_agg: pd.DataFrame) -> None:
    df = compare_agg.set_index("model").loc[["ours", "gatrust", "trustguard", "guardian"]].reset_index()
    labels = ["Ours", "GATrust-like", "TrustGuard-like", "Guardian-style"]
    colors = ["#0E7490", "#3B82F6", "#F59E0B", "#9CA3AF"]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
    metrics = [
        ("auc_pr_mean", "auc_pr_std", "AUC-PR", True),
        ("mcc_mean", "mcc_std", "MCC", True),
        ("ece_after_mean", "ece_after_std", "ECE", False),
    ]
    x = np.arange(len(df))
    for ax, (m, s, title, higher_better) in zip(axes, metrics):
        vals = df[m].to_numpy()
        errs = df[s].to_numpy()
        ax.bar(x, vals, yerr=errs, color=colors, edgecolor="#1F2937", linewidth=0.8, capsize=3)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        best = vals.argmax() if higher_better else vals.argmin()
        ax.bar(best, vals[best], color="#0F766E", edgecolor="#111827", linewidth=1.0)
    fig.suptitle("Main comparison under the unified temporal protocol", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figure2_main_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure2_main_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(repro_json: dict) -> None:
    model_order = ["ours", "gatrust", "trustguard", "guardian"]
    labels = {"ours": "Ours", "gatrust": "GATrust-like", "trustguard": "TrustGuard-like", "guardian": "Guardian-style"}
    colors = {"ours": "#0F766E", "gatrust": "#2563EB", "trustguard": "#EA580C", "guardian": "#6B7280"}
    by_model = {m: [] for m in model_order}
    for seed_blob in repro_json["seeds"]:
        for m in model_order:
            rc = seed_blob["models"][m]["risk_coverage"]
            by_model[m].append((np.array(rc["coverage"]), np.array(rc["risk"])))

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for m in model_order:
        coverages = by_model[m][0][0]
        risks = np.stack([r for _, r in by_model[m]], axis=0)
        mean = risks.mean(axis=0)
        std = risks.std(axis=0)
        ax.plot(coverages, mean, label=labels[m], color=colors[m], linewidth=2.2)
        ax.fill_between(coverages, mean - std, mean + std, color=colors[m], alpha=0.14)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title("Risk-coverage curves", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure3_risk_coverage.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure3_risk_coverage.svg", bbox_inches="tight")
    plt.close(fig)


def plot_decision_uncertainty(compare_agg: pd.DataFrame) -> None:
    df = compare_agg.set_index("model").loc[["ours", "gatrust", "trustguard", "guardian"]].reset_index()
    labels = ["Ours", "GATrust-like", "TrustGuard-like", "Guardian-style"]
    colors = ["#0F766E", "#2563EB", "#EA580C", "#6B7280"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    metrics = [
        ("rc_spearman_cov_risk_mean", "rc_spearman_cov_risk_std", "RC Spearman (higher is better)"),
        ("filt20_risk_mean", "filt20_risk_std", "Filtered Risk @20% (lower is better)"),
    ]
    x = np.arange(len(df))
    for ax, (m, s, title) in zip(axes, metrics):
        ax.bar(x, df[m], yerr=df[s], color=colors, edgecolor="#1F2937", linewidth=0.8, capsize=3)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("Decision-oriented uncertainty diagnostics", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figure4_decision_uncertainty.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure4_decision_uncertainty.svg", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation_agg: pd.DataFrame) -> None:
    df = ablation_agg.set_index("model").loc[["ours", "ours_nounc", "ours_1hop", "ours_notemp"]].reset_index()
    labels = ["Full", "w/o uncert.", "w/o multi-hop", "w/o temporal"]
    colors = ["#0F766E", "#7C3AED", "#2563EB", "#F59E0B"]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8))
    metrics = [
        ("auc_pr_mean", "auc_pr_std", "AUC-PR"),
        ("mcc_mean", "mcc_std", "MCC"),
        ("ece_after_mean", "ece_after_std", "ECE"),
    ]
    x = np.arange(len(df))
    for ax, (m, s, title) in zip(axes, metrics):
        ax.bar(x, df[m], yerr=df[s], color=colors, edgecolor="#1F2937", linewidth=0.8, capsize=3)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("Ablation study", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figure5_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure5_ablation.svg", bbox_inches="tight")
    plt.close(fig)


def write_prompts_and_audit() -> None:
    prompt = """Create a publication-quality system model figure for an IEEE-style paper on temporal signed trust prediction.\n\nVisual style:\n- clean white background\n- vector-diagram style\n- restrained academic palette (blue, teal, orange, light purple, gray)\n- no 3D effects, no cartoon icons, no decorative gradients\n- typography suitable for a journal figure\n- horizontal left-to-right pipeline\n- aspect ratio around 16:8\n\nRequired blocks from left to right:\n1. Temporal signed prefix graph\n   - show a historical directed signed graph\n   - indicate positive edges and negative edges separately\n   - annotate that only the historical prefix graph is visible during training\n   - include a small timeline: train prefix -> validation -> test\n2. Signed multi-hop encoder\n   - show separate positive and negative propagation channels\n   - include formulas: H^(0)=phi(XW_in+b), M^{+,k}=A~+ H^(k-1), M^{-,k}=A~- H^(k-1)\n   - emphasize multi-hop representations H^(0), H^(1), ..., H^(K)\n3. Temporal edge context\n   - show a candidate edge (u,v,t)\n   - show a small feature vector g_uv(t)=[delta_u, delta_v, c_uv^+, c_uv^-, delta_uv]\n   - label these as recency and pair-history statistics\n4. Edge-level evidential decoder\n   - show edge feature construction from node embeddings and temporal context\n   - include a compact formula for z_uv^(k)\n   - show output as a nonnegative evidence vector e_uv^(k)\n5. Global multi-hop evidence fusion\n   - show hop-wise evidence vectors fused by globally shared coefficients pi_k\n   - include formulas: pi_k = exp(beta_k)/sum_j exp(beta_j), e_uv = sum_k pi_k e_uv^(k)\n   - explicitly note that the hop coefficients are globally shared, not edge-specific attention\n6. Probability and uncertainty output\n   - show final Dirichlet parameter alpha_uv = e_uv + 1\n   - show outputs: trust probability p(y_uv=1|u,v,t) and uncertainty U_uv\n   - include a compact uncertainty formula U_uv = lambda U_vac + (1-lambda) H(p)\n   - add a short note that uncertainty supports selective prediction and risk-aware filtering\n\nFigure caption intent:\nA system overview of the proposed uncertainty-aware temporal trust prediction framework with prefix-only signed propagation, edge-level evidential decoding, and global multi-hop evidence fusion.\n"""
    (OUT / "figure1_system_model_prompt.txt").write_text(prompt, encoding="utf-8")

    audit = """PPT figure audit\n\nUsable with minor revision:\n- image16.png: visually polished multi-hop/evidence-fusion schematic, but not directly usable because it depicts unsigned propagation (A^k X) and labels the fusion as learnable attention, while the final method uses signed propagation and globally shared hop coefficients.\n\nNot suitable as final paper figures:\n- image32.png: uncertainty evaluation slide graphic is presentation-style and uses metrics/values from an older protocol, not the final unified results.\n- slides 8, 9, and 11: tables correspond to an earlier experiment version and conflict with the final unified `paper_best` results.\n- most other media files are icons or decorative assets rather than publication-ready figures.\n\nRecommendation:\n- redraw Figure 1 from scratch using the final method definition.\n- redraw all result plots from `paper_best_*` and `paper_assets_repro.json`.\n"""
    (OUT / "ppt_figure_audit.md").write_text(audit, encoding="utf-8")

    captions = """Suggested figure/table mapping for the paper\n\nFigure 1. System overview of the proposed uncertainty-aware temporal trust prediction framework.\nFigure 2. Main comparison under the unified temporal protocol (AUC-PR, MCC, ECE).\nFigure 3. Risk-coverage curves under the same best configuration.\nFigure 4. Decision-oriented uncertainty diagnostics (RC Spearman and filtered risk).\nFigure 5. Ablation study on uncertainty, multi-hop propagation, and temporal context.\nTable 1. Dataset statistics of the Bitcoin-OTC temporal signed trust prediction task.\nTable 2. Main comparison with protocol-aligned baselines.\nTable 3. Ablation results of the proposed model.\n"""
    (OUT / "asset_manifest.md").write_text(captions, encoding="utf-8")


def main():
    compare_agg, ablation_agg, _, repro_json = load_data()
    build_tables(compare_agg, ablation_agg)
    plot_system_model()
    plot_main_comparison(compare_agg)
    plot_risk_coverage(repro_json)
    plot_decision_uncertainty(compare_agg)
    plot_ablation(ablation_agg)
    write_prompts_and_audit()
    print(f"paper assets written to {OUT}")


if __name__ == "__main__":
    main()
