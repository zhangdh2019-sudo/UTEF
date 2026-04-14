import argparse
import json
import os
import csv
from typing import Dict, List, Tuple

import numpy as np
import torch

from data.temporal_split import build_temporal_dataset
from data.features import build_node_features
from model_link import TemporalSignedTrustPredictor
from train_link import train_link
from eval_link import evaluate_link_predictions
from utils import set_seeds

from baselines.heuristic_trust import fit_heuristic_trust
from baselines.embedding_baselines import (
    fit_node2vec_edge_classifier,
    fit_signed_node_embedding_edge_classifier,
    fit_sdne_like_embedding,
)
from baselines.gnn_edge_models import fit_torch_edge_classifier
from baselines.tgat import fit_tgat_like
from baselines.ga_trust import fit_ga_trust
from baselines.uncertainty_models import fit_evidential_mlp
from baselines.uncertainty_models import predict_mc_dropout_dirichlet
from baselines.guardian import fit_guardian
from baselines.trustguard_like import fit_trustguard_like
from consistency_checks import (
    check_feature_type_constraints,
    check_label_matches_rating,
    check_temporal_boundaries,
    check_threshold_freeze,
)


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    import numpy as np

    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def _save_rows_to_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _uncertainty_evidence_payload(r: Dict[str, object]) -> Dict[str, object]:
    """Serializable subset for risk–coverage plots and filtering tables."""
    rc = r["risk_coverage"]

    def _num(x) -> object:
        if isinstance(x, (np.integer, int)):
            return int(x)
        xf = float(x)
        return None if xf != xf else xf  # NaN -> null in JSON

    return {
        "risk_coverage": {
            "coverage": np.asarray(rc["coverage"], dtype=float).tolist(),
            "risk": np.asarray(rc["risk"], dtype=float).tolist(),
            "accuracy": np.asarray(rc["accuracy"], dtype=float).tolist(),
            "aurc": float(rc["aurc"]),
            "eaurc": float(rc["eaurc"]),
        },
        "risk_coverage_confidence": {
            "coverage": np.asarray(r["risk_coverage_confidence"]["coverage"], dtype=float).tolist(),
            "risk": np.asarray(r["risk_coverage_confidence"]["risk"], dtype=float).tolist(),
            "accuracy": np.asarray(r["risk_coverage_confidence"]["accuracy"], dtype=float).tolist(),
            "aurc": float(r["risk_coverage_confidence"]["aurc"]),
            "eaurc": float(r["risk_coverage_confidence"]["eaurc"]),
        },
        "risk_coverage_model_uncertainty": {
            "coverage": np.asarray(r["risk_coverage_model_uncertainty"]["coverage"], dtype=float).tolist(),
            "risk": np.asarray(r["risk_coverage_model_uncertainty"]["risk"], dtype=float).tolist(),
            "accuracy": np.asarray(r["risk_coverage_model_uncertainty"]["accuracy"], dtype=float).tolist(),
            "aurc": float(r["risk_coverage_model_uncertainty"]["aurc"]),
            "eaurc": float(r["risk_coverage_model_uncertainty"]["eaurc"]),
        },
        "risk_coverage_summary": {k: _num(v) for k, v in r["risk_coverage_summary"].items()},
        "risk_coverage_summary_confidence": {
            k: _num(v) for k, v in r["risk_coverage_summary_confidence"].items()
        },
        "risk_coverage_summary_model_uncertainty": {
            k: _num(v) for k, v in r["risk_coverage_summary_model_uncertainty"].items()
        },
        "selective": {k: _num(v) for k, v in r["selective"].items()},
        "selective_confidence": {k: _num(v) for k, v in r["selective_confidence"].items()},
        "selective_model_uncertainty": {k: _num(v) for k, v in r["selective_model_uncertainty"].items()},
        "uncertainty_filtering": {
            k: {kk: _num(vv) for kk, vv in v.items()} for k, v in r["uncertainty_filtering"].items()
        },
        "uncertainty_filtering_confidence": {
            k: {kk: _num(vv) for kk, vv in v.items()}
            for k, v in r["uncertainty_filtering_confidence"].items()
        },
        "uncertainty_filtering_model_uncertainty": {
            k: {kk: _num(vv) for kk, vv in v.items()}
            for k, v in r["uncertainty_filtering_model_uncertainty"].items()
        },
        "best_threshold": float(r["best_threshold"]),
    }


def run_one_seed(
    seed: int,
    args,
):
    set_seeds(seed)

    split = build_temporal_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
    )
    device = args.device

    features = build_node_features(
        split,
        feature_type=args.feature_type,
        node2vec_dim=args.node2vec_dim,
        node2vec_epochs=args.node2vec_epochs,
        device=device,
        node2vec_random_seed=seed,
    )

    # Minimal consistency checks (fail-fast during development).
    check_temporal_boundaries(split)
    check_label_matches_rating(split)
    expected_dim = {
        "minimal": 4,
        "signed_structural": 14,
        "node2vec": int(args.node2vec_dim),
    }[args.feature_type.lower()]
    check_feature_type_constraints(args.feature_type, features, expected_dim=expected_dim)

    input_dim = features.feature_dim

    _sel_kw = {
        "threshold_metric": args.threshold_metric,
        "do_temperature_scaling": True,
        "selective_uncertainty": args.selective_uncertainty,
        "selective_hybrid_weight": float(args.selective_hybrid_weight),
    }

    ours_variants = ("ours", "ours_nounc", "ours_1hop", "ours_notemp")
    requested_ours = [v for v in ours_variants if v in args.models]
    trained_ours_models = {}

    def train_ours_variant(variant: str):
        num_hops = 1 if variant == "ours_1hop" else int(args.num_hops)
        use_temporal = False if variant == "ours_notemp" else True
        lam_u = 0.0 if variant == "ours_nounc" else float(args.lambda_uncertainty)
        m = TemporalSignedTrustPredictor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_hops=num_hops,
            kl=args.kl,
            dis=args.dis,
            dropout=args.dropout,
            use_temporal_features=use_temporal,
            u_alpha=float(args.u_alpha),
        )
        m.build_temporal_stats(split)
        m, _ = train_link(
            model=m,
            split=split,
            features=features,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_uncertainty=lam_u,
            classification_loss=args.classification_loss,
            focal_gamma=args.focal_gamma,
            label_smoothing=float(args.label_smoothing),
            lambda_ranking=float(args.lambda_ranking),
            device=device,
        )
        trained_ours_models[variant] = m
        return m

    # Evaluate helpers
    def eval_dirichlet(model):
        model.eval()
        A_pos = split.A_pos.to(device)
        A_neg = split.A_neg.to(device)
        x = features.x.to(device)
        val_probs, val_u = model.predict_proba(
            x,
            A_pos,
            A_neg,
            split.val_edge_index.to(device),
            edge_timestamp=split.val_timestamp.to(device),
        )
        test_probs, test_u = model.predict_proba(
            x,
            A_pos,
            A_neg,
            split.test_edge_index.to(device),
            edge_timestamp=split.test_timestamp.to(device),
        )
        return evaluate_link_predictions(
            split=split,
            val_probs=val_probs.to(device),
            test_probs=test_probs.to(device),
            val_uncertainty=val_u.to(device),
            test_uncertainty=test_u.to(device),
            val_y=split.val_edge_label.to(device),
            test_y=split.test_edge_label.to(device),
            **_sel_kw,
        )

    results = {}

    for ov in requested_ours:
        results[ov] = eval_dirichlet(train_ours_variant(ov))

    # HeuristicTrust
    if "heuristic" in args.models:
        h_model = fit_heuristic_trust(split, device="cpu")
        val_probs, val_u = h_model.predict_proba(split.val_edge_index)
        test_probs, test_u = h_model.predict_proba(split.test_edge_index)
        results["heuristic"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    # Node2Vec / SiNE-like / SDNE-like
    if "node2vec" in args.models:
        m = fit_node2vec_edge_classifier(split, embedding_dim=args.node2vec_dim, device="cpu")
        val_probs, val_u = m.predict_proba(split.val_edge_index)
        test_probs, test_u = m.predict_proba(split.test_edge_index)
        results["node2vec"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    if "sine" in args.models:
        m = fit_signed_node_embedding_edge_classifier(split, embedding_dim=args.node2vec_dim, device="cpu")
        val_probs, val_u = m.predict_proba(split.val_edge_index)
        test_probs, test_u = m.predict_proba(split.test_edge_index)
        results["sine"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    if "sdne" in args.models:
        m = fit_sdne_like_embedding(split, dim=max(16, args.node2vec_dim // 2), epochs=50, device="cpu")
        val_probs, val_u = m.predict_proba(split.val_edge_index)
        test_probs, test_u = m.predict_proba(split.test_edge_index)
        results["sdne"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    # Structural GNN edge classifier baselines
    if "mlp" in args.models or "node2vec_mlp" in args.models:
        m = fit_torch_edge_classifier(split, features, model_type="mlp", hidden_dim=args.hidden_dim, device=device)
        val_probs, val_u = m.predict_proba(split.val_edge_index)
        test_probs, test_u = m.predict_proba(split.test_edge_index)
        results["node2vec_mlp"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    if "gcn" in args.models:
        m = fit_torch_edge_classifier(split, features, model_type="gcn", hidden_dim=args.hidden_dim, device=device)
        val_probs, val_u = m.predict_proba(split.val_edge_index)
        test_probs, test_u = m.predict_proba(split.test_edge_index)
        results["gcn"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    if "sage" in args.models:
        m = fit_torch_edge_classifier(split, features, model_type="sage", hidden_dim=args.hidden_dim, device=device)
        val_probs, val_u = m.predict_proba(split.val_edge_index)
        test_probs, test_u = m.predict_proba(split.test_edge_index)
        results["sage"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    if "gat" in args.models:
        m = fit_torch_edge_classifier(split, features, model_type="gat", hidden_dim=args.hidden_dim, device=device)
        val_probs, val_u = m.predict_proba(split.val_edge_index)
        test_probs, test_u = m.predict_proba(split.test_edge_index)
        results["gat"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    if "signed_gcn" in args.models:
        m = fit_torch_edge_classifier(split, features, model_type="signed_gcn", hidden_dim=args.hidden_dim, device=device)
        val_probs, val_u = m.predict_proba(split.val_edge_index)
        test_probs, test_u = m.predict_proba(split.test_edge_index)
        results["signed_gcn"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    # Temporal baseline TGAT
    if "tgat" in args.models:
        tgat = fit_tgat_like(
            split,
            hidden_dim=args.hidden_dim,
            k_history=args.k_history,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=min(args.epochs, 80),
            patience=args.patience,
            device=device,
        )
        val_probs, val_u = tgat.predict_proba(split.val_edge_index, split.val_timestamp)
        test_probs, test_u = tgat.predict_proba(split.test_edge_index, split.test_timestamp)
        results["tgat"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    # GAtrust
    if "gatrust" in args.models:
        ga = fit_ga_trust(
            split=split,
            features=features,
            hidden_dim=args.hidden_dim,
            num_hops=args.num_hops,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        val_probs, val_u = ga.predict_proba(split.val_edge_index)
        test_probs, test_u = ga.predict_proba(split.test_edge_index)
        results["gatrust"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    # TrustGuard-style temporal signed baseline (reimplementation; see baselines/trustguard_like.py)
    if "trustguard" in args.models:
        tg = fit_trustguard_like(
            split=split,
            features=features,
            hidden_dim=args.hidden_dim,
            num_hops=args.num_hops,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        val_probs, val_u = tg.predict_proba(split.val_edge_index, split.val_timestamp)
        test_probs, test_u = tg.predict_proba(split.test_edge_index, split.test_timestamp)
        results["trustguard"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    if "guardian" in args.models:
        gu = fit_guardian(split=split, features=features, device="cpu")
        val_probs, val_u = gu.predict_proba(split.val_edge_index)
        test_probs, test_u = gu.predict_proba(split.test_edge_index)
        results["guardian"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    if "mc_dropout" in args.models:
        if "ours" not in trained_ours_models:
            train_ours_variant("ours")
        mc_model = trained_ours_models["ours"]
        x_dev = features.x.to(device)
        A_pos = split.A_pos.to(device)
        A_neg = split.A_neg.to(device)
        val_mc = predict_mc_dropout_dirichlet(
            mc_model,
            x=x_dev,
            A_pos=A_pos,
            A_neg=A_neg,
            edge_index=split.val_edge_index.to(device),
            edge_timestamp=split.val_timestamp.to(device),
            mc_samples=20,
        )
        test_mc = predict_mc_dropout_dirichlet(
            mc_model,
            x=x_dev,
            A_pos=A_pos,
            A_neg=A_neg,
            edge_index=split.test_edge_index.to(device),
            edge_timestamp=split.test_timestamp.to(device),
            mc_samples=20,
        )
        results["mc_dropout"] = evaluate_link_predictions(
            split=split,
            val_probs=val_mc.probs,
            test_probs=test_mc.probs,
            val_uncertainty=val_mc.uncertainty,
            test_uncertainty=test_mc.uncertainty,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    # Uncertainty evidential MLP baseline (no graph propagation)
    if "evidential_mlp" in args.models:
        evid_mlp = fit_evidential_mlp(
            split=split,
            features=features,
            hidden_dim=args.hidden_dim,
            kl=args.kl,
            dis=args.dis,
            dropout=args.dropout,
            epochs=min(args.epochs, 120),
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        val_probs, val_u = evid_mlp.predict_proba(split.val_edge_index.to(device), features.x.to(device))
        test_probs, test_u = evid_mlp.predict_proba(split.test_edge_index.to(device), features.x.to(device))
        results["evidential_mlp"] = evaluate_link_predictions(
            split=split,
            val_probs=val_probs,
            test_probs=test_probs,
            val_uncertainty=val_u,
            test_uncertainty=test_u,
            val_y=split.val_edge_label,
            test_y=split.test_edge_label,
            **_sel_kw,
        )

    for model_name, out in results.items():
        check_threshold_freeze(out)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument(
        "--dataset",
        type=str,
        default="bitcoin_otc",
        choices=["bitcoin_otc", "bitcoin_alpha", "wiki_rfa"],
        help="Temporal signed dataset to evaluate.",
    )
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--feature-type", type=str, default="node2vec", choices=["minimal", "signed_structural", "node2vec"])
    p.add_argument("--models", type=str, default="ours,heuristic,guardian,node2vec,mlp,sine,sdne,gcn,sage,gat,signed_gcn,tgat,gatrust,trustguard,evidential_mlp,mc_dropout")
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=2e-5)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-hops", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kl", type=float, default=0.0)
    p.add_argument("--dis", type=float, default=0.0)
    p.add_argument(
        "--lambda-uncertainty",
        type=float,
        default=0.02,
        help="Weight for evidential regularizer on training edges (ours family only).",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.03,
        help="Binary label smoothing for BCE/focal targets (ours training only); improves probability spread for TS/ECE.",
    )
    p.add_argument(
        "--u-alpha",
        type=float,
        default=0.45,
        help="Blend in ours uncertainty: u = u_alpha * u_dirichlet + (1-u_alpha) * entropy.",
    )
    p.add_argument(
        "--lambda-ranking",
        type=float,
        default=0.045,
        help="Pairwise ranking loss weight on trust logit (ours family); helps AUC-PR and often selective curves.",
    )

    p.add_argument("--threshold-metric", type=str, default="balanced_accuracy", choices=["mcc", "balanced_accuracy"])
    p.add_argument("--classification-loss", type=str, default="bce", choices=["bce", "focal"])
    p.add_argument("--focal-gamma", type=float, default=2.0)

    p.add_argument("--node2vec-dim", type=int, default=64)
    p.add_argument("--node2vec-epochs", type=int, default=30)
    p.add_argument("--k-history", type=int, default=20)

    p.add_argument(
        "--selective-uncertainty",
        type=str,
        default="rank_hybrid",
        choices=["model", "calibrated_entropy", "rank_hybrid", "confidence_maxprob", "confidence_margin"],
        help="How to sort edges for risk-coverage / filtering (default rank_hybrid aligns TS decisions with uncertainty).",
    )
    p.add_argument(
        "--selective-hybrid-weight",
        type=float,
        default=0.52,
        help="In rank_hybrid: weight on rank(model_u); 1-weight on rank(entropy(p_cal)).",
    )

    p.add_argument("--out-csv", type=str, default="outputs/link_results.csv")
    p.add_argument(
        "--export-uncertainty-json",
        type=str,
        default="",
        help="If set, write per-seed risk-coverage arrays + filtering table payload for plotting.",
    )
    args = p.parse_args()

    args.models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    all_seed_results = []
    for seed in range(1, args.seeds + 1):
        print(f"[seed {seed}] building split + features + training/eval...")
        res = run_one_seed(seed=seed, args=args)
        all_seed_results.append((seed, res))

    # Aggregate to table rows
    # We store both raw per-seed and aggregated mean/std.
    rows_raw = []
    rows_agg = []

    model_keys = sorted({k for _, res in all_seed_results for k in res.keys()})
    main_metrics = [
        "auc_pr",
        "mcc",
        "macro_f1",
        "specificity",
        "pred_pos_rate",
        "aurc",
        "eaurc",
        "ece_after",
        "ece_before",
        "brier",
        "sel_acc_70",
        "sel_acc_80",
        "sel_acc_90",
        "filt20_acc",
        "filt20_risk",
        "rc_violation_rate",
        "rc_spearman_cov_risk",
        "aurc_confidence",
        "eaurc_confidence",
        "sel_conf_acc_80",
        "filt20_acc_confidence",
        "filt20_risk_confidence",
        "rc_spearman_cov_risk_confidence",
        "aurc_model_unc",
        "rc_spearman_cov_risk_model_unc",
    ]

    uncertainty_json: Dict[str, object] = {"seeds": []}

    for model_name in model_keys:
        per_seed = {m: [] for m in main_metrics}
        for seed, res in all_seed_results:
            r = res.get(model_name)
            if r is None:
                continue
            sel = r["selective"]
            sel_conf = r["selective_confidence"]
            filt20 = r["uncertainty_filtering"]["remove_top_20"]
            filt20_conf = r["uncertainty_filtering_confidence"]["remove_top_20"]
            rcs = r["risk_coverage_summary"]
            rcs_conf = r["risk_coverage_summary_confidence"]
            rcs_model = r["risk_coverage_summary_model_uncertainty"]
            row = {
                "seed": seed,
                "dataset": args.dataset,
                "model": model_name,
                "auc_roc": r["test_auc"]["auc_roc"],
                "auc_pr": r["test_auc"]["auc_pr"],
                "mcc": r["test_metrics"]["mcc"],
                "macro_f1": r["test_metrics"]["macro_f1"],
                "specificity": r["test_metrics"]["specificity"],
                "balanced_accuracy": r["test_metrics"]["balanced_accuracy"],
                "pred_pos_rate": r["test_metrics"]["pred_pos_rate"],
                "aurc": r["risk_coverage"]["aurc"],
                "eaurc": r["risk_coverage"]["eaurc"],
                "ece_before": r["ece_before"],
                "ece_after": r["ece_after"],
                "brier": r["brier"],
                "sel_acc_70": sel["acc@70"],
                "sel_acc_80": sel["acc@80"],
                "sel_acc_90": sel["acc@90"],
                "filt10_acc": r["uncertainty_filtering"]["remove_top_10"]["accuracy"],
                "filt10_risk": r["uncertainty_filtering"]["remove_top_10"]["risk"],
                "filt20_acc": filt20["accuracy"],
                "filt20_risk": filt20["risk"],
                "filt20_retain": filt20["retain_pct"],
                "filt30_acc": r["uncertainty_filtering"]["remove_top_30"]["accuracy"],
                "filt30_risk": r["uncertainty_filtering"]["remove_top_30"]["risk"],
                "rc_violation_rate": rcs["rc_violation_rate"],
                "rc_spearman_cov_risk": rcs["rc_spearman_cov_risk"],
                "aurc_confidence": r["risk_coverage_confidence"]["aurc"],
                "eaurc_confidence": r["risk_coverage_confidence"]["eaurc"],
                "sel_conf_acc_80": sel_conf["acc@80"],
                "filt20_acc_confidence": filt20_conf["accuracy"],
                "filt20_risk_confidence": filt20_conf["risk"],
                "rc_spearman_cov_risk_confidence": rcs_conf["rc_spearman_cov_risk"],
                "aurc_model_unc": r["risk_coverage_model_uncertainty"]["aurc"],
                "rc_spearman_cov_risk_model_unc": rcs_model["rc_spearman_cov_risk"],
            }
            rows_raw.append(row)
            per_seed["auc_pr"].append(r["test_auc"]["auc_pr"])
            per_seed["mcc"].append(r["test_metrics"]["mcc"])
            per_seed["macro_f1"].append(r["test_metrics"]["macro_f1"])
            per_seed["specificity"].append(r["test_metrics"]["specificity"])
            per_seed["pred_pos_rate"].append(r["test_metrics"]["pred_pos_rate"])
            per_seed["aurc"].append(r["risk_coverage"]["aurc"])
            per_seed["eaurc"].append(r["risk_coverage"]["eaurc"])
            per_seed["ece_after"].append(r["ece_after"])
            per_seed["ece_before"].append(r["ece_before"])
            per_seed["brier"].append(r["brier"])
            per_seed["sel_acc_70"].append(sel["acc@70"])
            per_seed["sel_acc_80"].append(sel["acc@80"])
            per_seed["sel_acc_90"].append(sel["acc@90"])
            per_seed["filt20_acc"].append(filt20["accuracy"])
            per_seed["filt20_risk"].append(filt20["risk"])
            per_seed["rc_violation_rate"].append(rcs["rc_violation_rate"])
            per_seed["rc_spearman_cov_risk"].append(rcs["rc_spearman_cov_risk"])
            per_seed["aurc_confidence"].append(r["risk_coverage_confidence"]["aurc"])
            per_seed["eaurc_confidence"].append(r["risk_coverage_confidence"]["eaurc"])
            per_seed["sel_conf_acc_80"].append(sel_conf["acc@80"])
            per_seed["filt20_acc_confidence"].append(filt20_conf["accuracy"])
            per_seed["filt20_risk_confidence"].append(filt20_conf["risk"])
            per_seed["rc_spearman_cov_risk_confidence"].append(rcs_conf["rc_spearman_cov_risk"])
            per_seed["aurc_model_unc"].append(r["risk_coverage_model_uncertainty"]["aurc"])
            per_seed["rc_spearman_cov_risk_model_unc"].append(rcs_model["rc_spearman_cov_risk"])

        agg = {"dataset": args.dataset, "model": model_name}
        for mk in main_metrics:
            mean, std = _mean_std(per_seed[mk])
            agg[mk + "_mean"] = mean
            agg[mk + "_std"] = std
        rows_agg.append(agg)

    for seed, res in all_seed_results:
        entry = {"seed": seed, "models": {k: _uncertainty_evidence_payload(v) for k, v in res.items()}}
        uncertainty_json["seeds"].append(entry)

    # Save
    _save_rows_to_csv(args.out_csv, rows_raw)
    _save_rows_to_csv(os.path.splitext(args.out_csv)[0] + "_agg.csv", rows_agg)

    if getattr(args, "export_uncertainty_json", "").strip():
        out_j = args.export_uncertainty_json.strip()
        os.makedirs(os.path.dirname(out_j) or ".", exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as jf:
            json.dump(uncertainty_json, jf, indent=2)

    # Print summary table
    print("\n=== Main Table (mean±std) ===")
    for model_name in model_keys:
        # find first seed value list again from rows_agg
        a = next(r for r in rows_agg if r["model"] == model_name)
        def fmt(k):
            return f"{a[k+'_mean']:.4f}±{a[k+'_std']:.4f}"
        print(
            f"{model_name:>15} | AUC-PR {fmt('auc_pr'):>18} | MCC {fmt('mcc'):>14} | "
            f"MacroF1 {fmt('macro_f1'):>14} | Spec {fmt('specificity'):>14} | AURC {fmt('aurc'):>14} | "
            f"ECE {fmt('ece_after'):>14} | Brier {fmt('brier'):>14}"
        )


if __name__ == "__main__":
    main()

