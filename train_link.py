import os
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from utils import reg_loss


@torch.no_grad()
def _eval_auc_pr(
    model,
    x: torch.Tensor,
    A_pos: torch.Tensor,
    A_neg: torch.Tensor,
    edge_index: torch.Tensor,
    edge_timestamp: torch.Tensor,
    edge_label: torch.Tensor,
) -> float:
    model.eval()
    probs, _u = model.predict_proba(x, A_pos, A_neg, edge_index, edge_timestamp)
    # trust probability = class 1
    trust_prob = probs[:, 1].detach().cpu().numpy()
    y = edge_label.detach().cpu().numpy()
    return float(average_precision_score(y, trust_prob))


def train_link(
    model,
    split,
    features,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    pos_weight: Optional[float] = None,
    lambda_uncertainty: float = 0.01,
    lambda_ranking: float = 0.0,
    classification_loss: str = "bce",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    device: str = "cpu",
    checkpoint_path: Optional[str] = None,
) -> Tuple[object, Dict[str, float]]:
    """
    Edge-level training with early stopping on val AUC-PR.
    """
    device_t = torch.device(device)

    x = features.x.to(device_t)
    A_pos = split.A_pos.to(device_t)
    A_neg = split.A_neg.to(device_t)

    train_edge_index = split.train_edge_index.to(device_t)
    val_edge_index = split.val_edge_index.to(device_t)

    y_train = split.train_edge_label.to(device_t)
    y_val = split.val_edge_label.to(device_t)

    # Class imbalance handling (balanced inverse-frequency weights).
    # Note: in BitcoinOTC, trust(label=1) is majority and distrust(label=0) is minority.
    if pos_weight is None:
        num_pos = int((y_train == 1).sum().item())
        num_neg = int((y_train == 0).sum().item())
        if num_pos == 0 or num_neg == 0:
            w_pos, w_neg = 1.0, 1.0
        else:
            total = float(num_pos + num_neg)
            w_pos = total / (2.0 * float(num_pos))
            w_neg = total / (2.0 * float(num_neg))
    else:
        # Interpret pos_weight as the positive-class sample weight; negative weight = 1.
        w_pos = float(pos_weight)
        w_neg = 1.0

    w_pos_t = torch.tensor(w_pos, dtype=torch.float32, device=device_t)
    w_neg_t = torch.tensor(w_neg, dtype=torch.float32, device=device_t)

    model = model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc_pr = -1.0
    best_state = None
    patience_t = 0

    # Precompute weights per training edge label to reuse each epoch.
    edge_weights = torch.where(y_train == 1, w_pos_t, w_neg_t)

    for epoch in range(1, epochs + 1):
        model.train()
        out = model(x, A_pos, A_neg, train_edge_index, edge_timestamp=split.train_timestamp.to(device_t), y=None)
        evidence_a = out["evidence"]
        probs = out["probs"]
        trust_logits = out["trust_logit"]  # log-odds [E]

        # classification loss aligned with decision boundary metrics:
        # BCE over p(trust) with positive class weighting.
        p_trust = probs[:, 1].clamp(1e-12, 1 - 1e-12)
        y_f = y_train.float()
        if label_smoothing > 0:
            ls = float(label_smoothing)
            y_f = y_f * (1.0 - ls) + 0.5 * ls
        bce = F.binary_cross_entropy(p_trust, y_f, reduction="none")  # [E]
        if classification_loss == "focal":
            pt = torch.where(y_train == 1, p_trust, 1.0 - p_trust).clamp(1e-12, 1 - 1e-12)
            focal_factor = torch.pow(1.0 - pt, float(focal_gamma))
            cls_loss = (focal_factor * bce * edge_weights).mean()
        else:
            # default: class-balanced BCE
            cls_loss = (bce * edge_weights).mean()

        # evidential regularization as uncertainty regularizer
        uncertainty_reg = reg_loss(y_train, evidence_a, c=2, kl=model.kl, dis=model.dis)
        uncertainty_reg_loss = uncertainty_reg.mean()

        loss = cls_loss + float(lambda_uncertainty) * uncertainty_reg_loss

        # Pairwise ranking loss to directly improve ranking metrics (AUC-PR).
        # Sample pos/neg edges and enforce logit(pos) > logit(neg).
        pos_idx = torch.where(y_train == 1)[0]
        neg_idx = torch.where(y_train == 0)[0]
        if pos_idx.numel() > 0 and neg_idx.numel() > 0 and float(lambda_ranking) > 0.0:
            sample_size = min(1024, pos_idx.numel(), neg_idx.numel())
            pos_sel = pos_idx[torch.randint(0, pos_idx.numel(), (sample_size,), device=device_t)]
            neg_sel = neg_idx[torch.randint(0, neg_idx.numel(), (sample_size,), device=device_t)]
            pos_l = trust_logits[pos_sel]
            neg_l = trust_logits[neg_sel]
            ranking_loss = F.softplus(neg_l - pos_l).mean()
            loss = loss + float(lambda_ranking) * ranking_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Early stopping on val AUC-PR
        val_auc_pr = _eval_auc_pr(
            model=model,
            x=x,
            A_pos=A_pos,
            A_neg=A_neg,
            edge_index=val_edge_index,
            edge_timestamp=split.val_timestamp.to(device_t),
            edge_label=y_val,
        )

        improved = val_auc_pr > best_auc_pr + 1e-8
        if improved:
            best_auc_pr = val_auc_pr
            best_state = deepcopy(model.state_dict())
            patience_t = 0
            if checkpoint_path is not None:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "best_auc_pr": best_auc_pr,
                        "state_dict": best_state,
                        "w_pos": w_pos,
                        "w_neg": w_neg,
                    },
                    checkpoint_path,
                )
        else:
            patience_t += 1
            if patience_t >= patience:
                break

    if best_state is None:
        best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    metrics = {"best_val_auc_pr": float(best_auc_pr)}
    return model, metrics

