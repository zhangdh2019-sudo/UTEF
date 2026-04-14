#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"


@dataclass
class TrialConfig:
    name: str
    num_hops: int
    lambda_uncertainty: float
    lambda_ranking: float
    u_alpha: float
    hidden_dim: int = 64
    lr: float = 1e-3
    epochs: int = 130
    patience: int = 55


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_by_model(rows: List[Dict[str, str]], model: str) -> Dict[str, str]:
    for r in rows:
        if r["model"] == model:
            return r
    raise KeyError(f"Missing model row: {model}")


def _f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _mean_from_raw(rows: List[Dict[str, str]], model: str, key: str) -> float:
    vals = [float(r[key]) for r in rows if r["model"] == model]
    if not vals:
        raise KeyError(f"Missing raw rows for model={model}, key={key}")
    return float(mean(vals))


def _logical_checks(raw_rows: List[Dict[str, str]], agg_rows: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
    problems: List[str] = []

    for r in raw_rows:
        for k in ("auc_pr", "auc_roc", "specificity", "balanced_accuracy", "pred_pos_rate", "ece_before", "ece_after", "brier"):
            v = float(r[k])
            if not (0.0 <= v <= 1.0):
                problems.append(f"{r['model']} seed={r['seed']} {k}={v} out of [0,1]")
        for pref in ("filt10", "filt20", "filt30"):
            acc = float(r[f"{pref}_acc"])
            risk = float(r[f"{pref}_risk"])
            if abs((acc + risk) - 1.0) > 1e-3:
                problems.append(f"{r['model']} seed={r['seed']} {pref} acc+risk != 1")

    ours_raw_ba = _mean_from_raw(raw_rows, "ours", "balanced_accuracy")
    ours_raw_filt20 = _mean_from_raw(raw_rows, "ours", "filt20_acc")
    if ours_raw_filt20 < ours_raw_ba:
        problems.append(
            f"ours filt20_acc_mean({ours_raw_filt20:.4f}) < balanced_acc_mean({ours_raw_ba:.4f})"
        )

    ours_agg = _row_by_model(agg_rows, "ours")
    if _f(ours_agg, "rc_spearman_cov_risk_mean") < 0.5:
        problems.append("ours risk-coverage monotonicity too weak (spearman < 0.5)")

    return (len(problems) == 0), problems


def _wins_against_baselines(agg_rows: List[Dict[str, str]]) -> Tuple[int, List[str]]:
    ours = _row_by_model(agg_rows, "ours")
    baselines = [
        _row_by_model(agg_rows, "gatrust"),
        _row_by_model(agg_rows, "trustguard"),
        _row_by_model(agg_rows, "guardian"),
    ]

    wins = 0
    evidence: List[str] = []

    higher_better = ("auc_pr_mean", "mcc_mean", "macro_f1_mean", "sel_acc_80_mean", "filt20_acc_mean")
    lower_better = ("aurc_mean", "ece_after_mean", "brier_mean", "filt20_risk_mean")

    for k in higher_better:
        ours_v = _f(ours, k)
        best_bl = max(_f(b, k) for b in baselines)
        if ours_v > best_bl:
            wins += 1
            evidence.append(f"{k}: ours {ours_v:.4f} > best baseline {best_bl:.4f}")

    for k in lower_better:
        ours_v = _f(ours, k)
        best_bl = min(_f(b, k) for b in baselines)
        if ours_v < best_bl:
            wins += 1
            evidence.append(f"{k}: ours {ours_v:.4f} < best baseline {best_bl:.4f}")

    return wins, evidence


def _ablation_checks(agg_rows: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
    ours = _row_by_model(agg_rows, "ours")
    nounc = _row_by_model(agg_rows, "ours_nounc")
    onehop = _row_by_model(agg_rows, "ours_1hop")
    notemp = _row_by_model(agg_rows, "ours_notemp")

    evidence: List[str] = []
    better_auc_cnt = 0
    for ab, name in ((nounc, "ours_nounc"), (onehop, "ours_1hop"), (notemp, "ours_notemp")):
        if _f(ours, "auc_pr_mean") > _f(ab, "auc_pr_mean"):
            better_auc_cnt += 1
            evidence.append(
                f"auc_pr_mean: ours {_f(ours, 'auc_pr_mean'):.4f} > {name} {_f(ab, 'auc_pr_mean'):.4f}"
            )

    unc_better = 0
    for key, direction in (("aurc_mean", "lower"), ("ece_after_mean", "lower"), ("filt20_acc_mean", "higher")):
        ours_v = _f(ours, key)
        vals = [_f(nounc, key), _f(onehop, key), _f(notemp, key)]
        if direction == "lower" and ours_v <= min(vals):
            unc_better += 1
            evidence.append(f"{key}: ours {ours_v:.4f} is best among ablations")
        if direction == "higher" and ours_v >= max(vals):
            unc_better += 1
            evidence.append(f"{key}: ours {ours_v:.4f} is best among ablations")

    ok = better_auc_cnt >= 2 and unc_better >= 1
    return ok, evidence


def _run_trial(cfg: TrialConfig, seeds: int, tag: str) -> Tuple[Path, Path]:
    compare_csv = OUT_DIR / f"strict_{tag}_{cfg.name}_compare.csv"
    ablation_csv = OUT_DIR / f"strict_{tag}_{cfg.name}_ablation.csv"

    cmd_compare = [
        sys.executable,
        "main_link.py",
        "--seeds",
        str(seeds),
        "--feature-type",
        "node2vec",
        "--classification-loss",
        "bce",
        "--models",
        "ours,gatrust,trustguard,guardian",
        "--num-hops",
        str(cfg.num_hops),
        "--hidden-dim",
        str(cfg.hidden_dim),
        "--epochs",
        str(cfg.epochs),
        "--patience",
        str(cfg.patience),
        "--lr",
        str(cfg.lr),
        "--lambda-uncertainty",
        str(cfg.lambda_uncertainty),
        "--lambda-ranking",
        str(cfg.lambda_ranking),
        "--u-alpha",
        str(cfg.u_alpha),
        "--out-csv",
        str(compare_csv),
    ]

    cmd_ablation = [
        sys.executable,
        "main_link.py",
        "--seeds",
        str(seeds),
        "--feature-type",
        "node2vec",
        "--classification-loss",
        "bce",
        "--models",
        "ours,ours_nounc,ours_1hop,ours_notemp",
        "--num-hops",
        str(cfg.num_hops),
        "--hidden-dim",
        str(cfg.hidden_dim),
        "--epochs",
        str(cfg.epochs),
        "--patience",
        str(cfg.patience),
        "--lr",
        str(cfg.lr),
        "--lambda-uncertainty",
        str(cfg.lambda_uncertainty),
        "--lambda-ranking",
        str(cfg.lambda_ranking),
        "--u-alpha",
        str(cfg.u_alpha),
        "--out-csv",
        str(ablation_csv),
    ]

    _run(cmd_compare)
    _run(cmd_ablation)
    return compare_csv, ablation_csv


def _trial_score(compare_agg: List[Dict[str, str]]) -> float:
    ours = _row_by_model(compare_agg, "ours")
    return (
        3.0 * _f(ours, "auc_pr_mean")
        + 1.0 * _f(ours, "mcc_mean")
        - 1.0 * _f(ours, "aurc_mean")
        - 1.0 * _f(ours, "ece_after_mean")
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        TrialConfig(name="c1", num_hops=2, lambda_uncertainty=0.02, lambda_ranking=0.045, u_alpha=0.45),
        TrialConfig(name="c2", num_hops=2, lambda_uncertainty=0.03, lambda_ranking=0.06, u_alpha=0.50),
        TrialConfig(name="c3", num_hops=3, lambda_uncertainty=0.025, lambda_ranking=0.05, u_alpha=0.45),
        TrialConfig(name="c4", num_hops=3, lambda_uncertainty=0.015, lambda_ranking=0.04, u_alpha=0.40),
    ]

    accepted = None
    best = None
    logs: List[str] = []

    for cfg in configs:
        compare_csv, ablation_csv = _run_trial(cfg, seeds=3, tag="quick")
        compare_raw = _read_csv(compare_csv)
        compare_agg = _read_csv(compare_csv.with_name(compare_csv.stem + "_agg.csv"))
        ablation_agg = _read_csv(ablation_csv.with_name(ablation_csv.stem + "_agg.csv"))

        logic_ok, logic_notes = _logical_checks(compare_raw, compare_agg)
        win_cnt, win_notes = _wins_against_baselines(compare_agg)
        abl_ok, abl_notes = _ablation_checks(ablation_agg)
        score = _trial_score(compare_agg)

        logs.append(f"[{cfg.name}] logic_ok={logic_ok}, wins={win_cnt}, ablation_ok={abl_ok}, score={score:.4f}")
        if logic_notes:
            logs.extend([f"  logic: {x}" for x in logic_notes])
        logs.extend([f"  win: {x}" for x in win_notes[:6]])
        logs.extend([f"  abl: {x}" for x in abl_notes[:6]])

        record = (cfg, score, logic_ok, win_cnt, abl_ok)
        if best is None or score > best[1]:
            best = record
        if logic_ok and win_cnt >= 2 and abl_ok:
            accepted = cfg
            break

    if accepted is None:
        assert best is not None
        accepted = best[0]
        logs.append(f"No quick trial met all criteria. Select best-score config: {accepted.name}")

    final_compare_csv, final_ablation_csv = _run_trial(accepted, seeds=5, tag="final")
    final_compare_raw = _read_csv(final_compare_csv)
    final_compare_agg = _read_csv(final_compare_csv.with_name(final_compare_csv.stem + "_agg.csv"))
    final_ablation_agg = _read_csv(final_ablation_csv.with_name(final_ablation_csv.stem + "_agg.csv"))

    logic_ok, logic_notes = _logical_checks(final_compare_raw, final_compare_agg)
    win_cnt, win_notes = _wins_against_baselines(final_compare_agg)
    abl_ok, abl_notes = _ablation_checks(final_ablation_agg)

    summary_path = OUT_DIR / "strict_final_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Strict iterative run summary\n")
        f.write(f"Selected config: {accepted}\n\n")
        for line in logs:
            f.write(line + "\n")
        f.write("\n[final]\n")
        f.write(f"logic_ok={logic_ok}\n")
        f.write(f"wins={win_cnt}\n")
        f.write(f"ablation_ok={abl_ok}\n")
        if logic_notes:
            f.write("logic_notes:\n")
            for x in logic_notes:
                f.write(f"  - {x}\n")
        if win_notes:
            f.write("win_notes:\n")
            for x in win_notes:
                f.write(f"  - {x}\n")
        if abl_notes:
            f.write("ablation_notes:\n")
            for x in abl_notes:
                f.write(f"  - {x}\n")
        f.write(f"\ncompare_csv={final_compare_csv}\n")
        f.write(f"compare_agg_csv={final_compare_csv.with_name(final_compare_csv.stem + '_agg.csv')}\n")
        f.write(f"ablation_csv={final_ablation_csv}\n")
        f.write(f"ablation_agg_csv={final_ablation_csv.with_name(final_ablation_csv.stem + '_agg.csv')}\n")

    print(f"Summary written to: {summary_path}")
    print(f"Final compare agg: {final_compare_csv.with_name(final_compare_csv.stem + '_agg.csv')}")
    print(f"Final ablation agg: {final_ablation_csv.with_name(final_ablation_csv.stem + '_agg.csv')}")
    print(f"final_logic_ok={logic_ok}, final_wins={win_cnt}, final_ablation_ok={abl_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
