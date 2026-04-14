from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    balanced_accuracy_score,
)


def _safe_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # confusion_matrix with fixed labels order [0,1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-12)
    accuracy = (tp + tn) / (tn + fp + fn + tp + 1e-12)
    return {
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def metrics_from_probs(
    y_true: np.ndarray,
    p_trust: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (p_trust >= threshold).astype(np.int64)
    cm = _safe_confusion_metrics(y_true, y_pred)

    out = {
        "accuracy": cm["accuracy"],
        "specificity": cm["specificity"],
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pred_pos_rate": float(y_pred.mean()),
    }
    return out


def auc_metrics(y_true: np.ndarray, p_trust: np.ndarray) -> Dict[str, float]:
    out = {
        "auc_roc": float(roc_auc_score(y_true, p_trust)),
        "auc_pr": float(average_precision_score(y_true, p_trust)),
    }
    return out


def find_best_threshold(
    y_true: np.ndarray,
    p_trust: np.ndarray,
    metric: str = "mcc",
    num_thresholds: int = 201,
) -> Tuple[float, Dict[str, float]]:
    """
    Scan thresholds in [0,1] to maximize the given metric.
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    best_thr = 0.5
    best_score = -1e9
    best_metrics = {}
    for thr in thresholds:
        m = metrics_from_probs(y_true, p_trust, threshold=float(thr))
        score = m["mcc"] if metric == "mcc" else m["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics


def brier_score(y_true: np.ndarray, p_trust: np.ndarray) -> float:
    y_true_f = y_true.astype(np.float64)
    return float(np.mean((p_trust - y_true_f) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    p_trust: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    ECE with confidence = max(p, 1-p).
    """
    y_true_t = y_true.astype(np.int64)
    p = p_trust.astype(np.float64)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(np.int64)
    acc = (pred == y_true_t).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_prop = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # include right edge for last bin
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        bin_prop[i] = mask.mean()
        bin_conf[i] = conf[mask].mean()
        bin_acc[i] = acc[mask].mean()
        ece += abs(bin_acc[i] - bin_conf[i]) * bin_prop[i]

    payload = {
        "bin_conf": bin_conf,
        "bin_acc": bin_acc,
        "bin_prop": bin_prop,
        "bins": bins,
    }
    return float(ece), payload


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


def _binary_entropy_np(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map values to [0,1] by rank (ties broken by mergesort order)."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n <= 1:
        return np.zeros_like(x, dtype=np.float64)
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / float(n - 1)


def combine_selective_uncertainty(
    u_model: np.ndarray,
    p_trust_cal: np.ndarray,
    mode: str = "model",
    hybrid_model_weight: float = 0.58,
) -> np.ndarray:
    """
    Selective-prediction sorting signal (risk-coverage, filtering).

    - model: use network-provided u only.
    - calibrated_entropy: entropy of temperature-scaled p (comparable to many baselines).
    - rank_hybrid: convex combination of rank(u) and rank(H(p_cal)) so sorting aligns
      with both evidential signal and post-calibration aleatoric uncertainty.
    - confidence_maxprob: u = 1 - max(p, 1-p) (confidence-based selective prediction).
    - confidence_margin: u = 0.5 - |p-0.5| (equivalent confidence ranking).
    """
    u_model = np.asarray(u_model, dtype=np.float64)
    if mode == "model":
        return u_model
    if mode == "calibrated_entropy":
        return _binary_entropy_np(p_trust_cal)
    if mode == "rank_hybrid":
        ru = _rank_normalize(u_model)
        re = _rank_normalize(_binary_entropy_np(p_trust_cal))
        w = float(hybrid_model_weight)
        return w * ru + (1.0 - w) * re
    p = np.asarray(p_trust_cal, dtype=np.float64)
    if mode == "confidence_maxprob":
        conf = np.maximum(p, 1.0 - p)
        return 1.0 - conf
    if mode == "confidence_margin":
        return 0.5 - np.abs(p - 0.5)
    raise ValueError(f"Unknown selective uncertainty mode: {mode}")


def _fit_ts_nll_once(
    y: torch.Tensor,
    logits: torch.Tensor,
    logT0: float,
    b0: float,
    max_iter: int,
    lr: float,
) -> Tuple[float, float, float, np.ndarray]:
    """One LBFGS run from (logT0, b0). Returns (nll, T, b, val_p_cal)."""
    logT = torch.nn.Parameter(torch.tensor(logT0, dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor(b0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS([logT, b], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(logT).clamp(1e-4, 1e4)
        logits_t = logits / T + b
        p_cal = torch.sigmoid(logits_t)
        eps = 1e-12
        nll = -(y * torch.log(p_cal + eps) + (1 - y) * torch.log(1 - p_cal + eps)).mean()
        nll.backward()
        return nll

    optimizer.step(closure)

    T = float(torch.exp(logT).detach().cpu().clamp(1e-4, 1e4).item())
    b_val = float(b.detach().cpu().item())
    with torch.no_grad():
        logits_t = logits / T + b_val
        p_cal_t = torch.sigmoid(logits_t)
        eps = 1e-12
        nll_final = float(
            -(y * torch.log(p_cal_t + eps) + (1 - y) * torch.log(1 - p_cal_t + eps)).mean().cpu().item()
        )
    p_cal = p_cal_t.detach().cpu().numpy()
    return nll_final, T, b_val, p_cal


def temperature_scale_binary(
    y_true: np.ndarray,
    p_trust: np.ndarray,
    max_iter: int = 2000,
    lr: float = 0.05,
) -> Tuple[float, float, np.ndarray]:
    """
    Fit temperature T (and bias b) on val by minimizing NLL.
    We apply: p' = sigmoid(logit(p)/T + b)

    Multi-restart LBFGS reduces sensitivity to poor local minima (important for stable ECE).
    """
    y = torch.tensor(y_true, dtype=torch.float64)
    logits = torch.tensor(_logit(p_trust), dtype=torch.float64)

    inits = [
        (0.0, 0.0),
        (float(np.log(2.0)), 0.0),
        (float(-np.log(2.0)), 0.0),
        (0.0, 0.5),
        (0.0, -0.5),
        (float(np.log(0.5)), -0.25),
    ]

    best_nll = float("inf")
    best_T, best_b, best_p = 1.0, 0.0, torch.sigmoid(logits).detach().cpu().numpy()

    for logT0, b0 in inits:
        nll, T, b_val, p_cal = _fit_ts_nll_once(y, logits, logT0, b0, max_iter=max_iter, lr=lr)
        if nll < best_nll:
            best_nll = nll
            best_T, best_b, best_p = T, b_val, p_cal

    return best_T, best_b, best_p


def risk_coverage_curve(
    y_true: np.ndarray,
    p_trust: np.ndarray,
    uncertainty: np.ndarray,
    num_points: int = 20,
    decision_threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    risk-coverage curve sorted by ascending uncertainty (low uncertainty kept first).
    """
    N = y_true.shape[0]
    order = np.argsort(uncertainty)  # low uncertainty first
    y_sorted = y_true[order]
    # Use the same decision threshold as classification metrics.
    pred_sorted = (p_trust[order] >= decision_threshold).astype(np.int64)
    correct = (pred_sorted == y_sorted).astype(np.float64)

    coverages = []
    risks = []
    accs = []
    for i in range(1, num_points + 1):
        k = int(np.ceil(N * (i / num_points)))
        k = max(1, min(k, N))
        cov = k / N
        acc_k = float(correct[:k].mean())
        risk_k = 1.0 - acc_k
        coverages.append(cov)
        risks.append(risk_k)
        accs.append(acc_k)

    coverages = np.asarray(coverages, dtype=np.float64)
    risks = np.asarray(risks, dtype=np.float64)
    accs = np.asarray(accs, dtype=np.float64)

    # AURC: average risk over curve points (discrete approximation)
    aurc = float(risks.mean())
    overall_risk = 1.0 - float(correct.mean())
    # Simple EAURC proxy: excess over overall risk baseline.
    eaurc = float(aurc - overall_risk)
    return {"coverage": coverages, "risk": risks, "accuracy": accs, "aurc": aurc, "eaurc": eaurc}


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    sx = rx.std()
    sy = ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def risk_coverage_curve_summary(rc: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    When sorting by ascending uncertainty, keeping only the most certain edges first:
    risk should generally *increase* as coverage increases (more edges included).
    We count violations of risk[i] > risk[i+1] + eps (non-monotone non-increasing risk).
    """
    risks = rc["risk"]
    if risks.size < 2:
        return {"rc_violations": 0.0, "rc_violation_rate": 0.0, "rc_spearman_cov_risk": float("nan")}
    eps = 1e-9
    violations = 0
    for i in range(len(risks) - 1):
        if risks[i + 1] + eps < risks[i]:
            violations += 1
    violation_rate = violations / max(1, (len(risks) - 1))
    cov = rc["coverage"]
    sp_val = _spearman_rho(cov, risks)
    return {
        "rc_violations": float(violations),
        "rc_violation_rate": float(violation_rate),
        "rc_spearman_cov_risk": sp_val,
    }


def selective_accuracy_at_coverages(
    y_true: np.ndarray,
    p_trust: np.ndarray,
    uncertainty: np.ndarray,
    coverages: List[float] = [0.9, 0.8, 0.7],
    decision_threshold: float = 0.5,
) -> Dict[str, float]:
    N = y_true.shape[0]
    order = np.argsort(uncertainty)  # low uncertainty first
    y_sorted = y_true[order]
    pred_sorted = (p_trust[order] >= decision_threshold).astype(np.int64)
    correct = (pred_sorted == y_sorted).astype(np.float64)

    out = {}
    for cov in coverages:
        k = int(np.ceil(N * cov))
        k = max(1, min(k, N))
        out[f"acc@{int(cov*100)}"] = float(correct[:k].mean())
    return out


def uncertainty_filtering(
    y_true: np.ndarray,
    p_trust: np.ndarray,
    uncertainty: np.ndarray,
    thresholds: List[float] = [0.1, 0.2, 0.3],
    decision_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Remove top q uncertainty (highest uncertainty), evaluate on remaining.
    thresholds are proportions to remove.
    """
    N = y_true.shape[0]
    order = np.argsort(uncertainty)  # low uncertainty first
    keep_order = order  # we'll keep first k after removing

    out = {}
    for q in thresholds:
        rm = int(np.ceil(N * q))
        k = max(1, N - rm)
        keep_idx = keep_order[:k]
        y_k = y_true[keep_idx]
        p_k = p_trust[keep_idx]

        metrics = metrics_from_probs(y_k, p_k, threshold=decision_threshold)
        auc = auc_metrics(y_k, p_k)
        retain_pct = float(k / N)
        risk_k = 1.0 - metrics["accuracy"]
        out[f"remove_top_{int(q*100)}"] = {
            **metrics,
            **auc,
            "retain_pct": retain_pct,
            "risk": float(risk_k),
        }
    return out


def evaluate_link_predictions(
    split,
    val_probs: torch.Tensor,
    test_probs: torch.Tensor,
    val_uncertainty: Optional[torch.Tensor],
    test_uncertainty: Optional[torch.Tensor],
    val_y: torch.Tensor,
    test_y: torch.Tensor,
    threshold_metric: str = "mcc",
    do_temperature_scaling: bool = True,
    n_bins_ece: int = 15,
    selective_uncertainty: str = "rank_hybrid",
    selective_hybrid_weight: float = 0.58,
) -> Dict[str, object]:
    """
    Generic evaluator given predicted probabilities and uncertainties for val/test edges.
    """
    val_p = val_probs[:, 1].detach().cpu().numpy()
    test_p = test_probs[:, 1].detach().cpu().numpy()
    val_y_np = val_y.detach().cpu().numpy().astype(np.int64)
    test_y_np = test_y.detach().cpu().numpy().astype(np.int64)

    # Temperature scaling (binary) fitted on val.
    temperature = None
    temperature_bias = None
    val_p_cal = val_p
    test_p_cal = test_p
    ece_before = None
    ece_after = None

    if do_temperature_scaling:
        T, b_val, val_p_cal = temperature_scale_binary(val_y_np, val_p)
        temperature = T
        temperature_bias = float(b_val)
        # Apply same T to test
        test_logits = torch.tensor(_logit(test_p), dtype=torch.float64)
        test_p_cal = torch.sigmoid(test_logits / float(T) + float(b_val)).detach().cpu().numpy()

    # Threshold selection on calibrated val probabilities (if TS enabled)
    best_thr, val_thr_metrics = find_best_threshold(val_y_np, val_p_cal, metric=threshold_metric)

    # Test metrics
    test_auc = auc_metrics(test_y_np, test_p_cal)
    test_pred_metrics = metrics_from_probs(test_y_np, test_p_cal, threshold=best_thr)

    # Calibration metrics on test
    ece_before, rel_payload_before = expected_calibration_error(test_y_np, test_p, n_bins=n_bins_ece)
    ece_after, rel_payload_after = expected_calibration_error(test_y_np, test_p_cal, n_bins=n_bins_ece)
    brier = brier_score(test_y_np, test_p_cal)

    # Risk-coverage & selective prediction
    if test_uncertainty is None:
        raise ValueError("uncertainty is required for risk-coverage and selective prediction")
    test_unc_np = test_uncertainty.detach().cpu().numpy()
    unc_test = combine_selective_uncertainty(
        test_unc_np,
        test_p_cal,
        mode=selective_uncertainty,
        hybrid_model_weight=selective_hybrid_weight,
    )
    unc_conf = combine_selective_uncertainty(test_unc_np, test_p_cal, mode="confidence_maxprob")
    unc_model = combine_selective_uncertainty(test_unc_np, test_p_cal, mode="model")
    rc = risk_coverage_curve(
        test_y_np,
        test_p_cal,
        uncertainty=unc_test,
        decision_threshold=best_thr,
    )
    selective = selective_accuracy_at_coverages(
        test_y_np,
        test_p_cal,
        uncertainty=unc_test,
        decision_threshold=best_thr,
    )

    filtering = uncertainty_filtering(
        y_true=test_y_np,
        p_trust=test_p_cal,
        uncertainty=unc_test,
        thresholds=[0.1, 0.2, 0.3],
        decision_threshold=best_thr,
    )
    rc_conf = risk_coverage_curve(
        test_y_np,
        test_p_cal,
        uncertainty=unc_conf,
        decision_threshold=best_thr,
    )
    sel_conf = selective_accuracy_at_coverages(
        test_y_np,
        test_p_cal,
        uncertainty=unc_conf,
        decision_threshold=best_thr,
    )
    filt_conf = uncertainty_filtering(
        y_true=test_y_np,
        p_trust=test_p_cal,
        uncertainty=unc_conf,
        thresholds=[0.1, 0.2, 0.3],
        decision_threshold=best_thr,
    )

    rc_model = risk_coverage_curve(
        test_y_np,
        test_p_cal,
        uncertainty=unc_model,
        decision_threshold=best_thr,
    )
    sel_model = selective_accuracy_at_coverages(
        test_y_np,
        test_p_cal,
        uncertainty=unc_model,
        decision_threshold=best_thr,
    )
    filt_model = uncertainty_filtering(
        y_true=test_y_np,
        p_trust=test_p_cal,
        uncertainty=unc_model,
        thresholds=[0.1, 0.2, 0.3],
        decision_threshold=best_thr,
    )

    rc_summary = risk_coverage_curve_summary(rc)
    rc_summary_conf = risk_coverage_curve_summary(rc_conf)
    rc_summary_model = risk_coverage_curve_summary(rc_model)

    # OOD breakdown: endpoints seen in train?
    seen_mask = split.seen_node_mask.detach().cpu().numpy() if hasattr(split, "seen_node_mask") else None
    ood_results = {}
    if seen_mask is not None:
        test_edge_index = split.test_edge_index.detach().cpu()
        u = test_edge_index[0].numpy()
        v = test_edge_index[1].numpy()
        seen_u = seen_mask[u]
        seen_v = seen_mask[v]
        cat = np.where(seen_u & seen_v, 0, np.where(seen_u ^ seen_v, 1, 2))
        # 0 Seen-Seen, 1 Seen-Unseen, 2 Unseen-Unseen
        for c_id, name in [(0, "seen_seen"), (1, "seen_unseen"), (2, "unseen_unseen")]:
            mask = cat == c_id
            if mask.sum() == 0:
                continue
            auc_c = auc_metrics(test_y_np[mask], test_p_cal[mask])
            pred_m = metrics_from_probs(test_y_np[mask], test_p_cal[mask], threshold=best_thr)
            ood_results[name] = {**auc_c, **pred_m}

    return {
        "temperature": temperature,
        "temperature_bias": temperature_bias,
        "selective_uncertainty": selective_uncertainty,
        "best_threshold": float(best_thr),
        "val_thr_metrics": {k: float(v) for k, v in val_thr_metrics.items()},
        "test_auc": test_auc,
        "test_metrics": test_pred_metrics,
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "brier": float(brier),
        "reliability_payload_before": rel_payload_before,
        "reliability_payload_after": rel_payload_after,
        "risk_coverage": rc,
        "risk_coverage_summary": rc_summary,
        "selective": selective,
        "uncertainty_filtering": filtering,
        # Secondary selective protocols for reviewer-facing comparisons.
        "risk_coverage_confidence": rc_conf,
        "risk_coverage_summary_confidence": rc_summary_conf,
        "selective_confidence": sel_conf,
        "uncertainty_filtering_confidence": filt_conf,
        "risk_coverage_model_uncertainty": rc_model,
        "risk_coverage_summary_model_uncertainty": rc_summary_model,
        "selective_model_uncertainty": sel_model,
        "uncertainty_filtering_model_uncertainty": filt_model,
        "ood_breakdown": ood_results,
    }

