from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from data.temporal_split import TemporalSignedSplit
from data.features import FeaturePack


def _entropy_from_p(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


class GAtrustLike(nn.Module):
    """
    GAtrust-like signed trust propagation with gating.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_hops: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()

        self.W_gate = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.W_out = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hops)])

        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, A_pos: torch.Tensor, A_neg: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(x))
        h = self.dropout(h)

        for hop in range(self.num_hops):
            hp = torch.sparse.mm(A_pos, h)
            hn = torch.sparse.mm(A_neg, h)
            gate = torch.sigmoid(self.W_gate[hop](torch.cat([hp, hn], dim=1)))
            h = gate * hp + (1.0 - gate) * hn
            h = self.act(self.W_out[hop](h))
            h = self.dropout(h)

        u = edge_index[0]
        v = edge_index[1]
        hu = h[u]
        hv = h[v]
        feat = torch.cat([hu, hv, torch.abs(hu - hv), hu * hv], dim=1)
        logits = self.edge_mlp(feat).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        A_pos: torch.Tensor,
        A_neg: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        logits = self.forward(x, A_pos, A_neg, edge_index)
        p = torch.sigmoid(logits)
        probs = torch.stack([1 - p, p], dim=1)
        ent = _entropy_from_p(p)
        return probs.detach().cpu(), ent.detach().cpu()


@dataclass
class GATrustBaseline:
    model: GAtrustLike
    x: torch.Tensor
    A_pos: torch.Tensor
    A_neg: torch.Tensor
    device: str

    def predict_proba(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = edge_index.to(self.device)
        return self.model.predict_proba(self.x, self.A_pos, self.A_neg, edge_index)


def fit_ga_trust(
    split: TemporalSignedSplit,
    features: FeaturePack,
    hidden_dim: int = 64,
    num_hops: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    patience: int = 30,
    device: str = "cpu",
) -> GATrustBaseline:
    """
    GAtrust baseline: signed trust propagation with gating, then edge-level binary classification.
    """
    device_t = torch.device(device)
    x = features.x.to(device_t)
    A_pos = split.A_pos.to(device_t)
    A_neg = split.A_neg.to(device_t)

    train_edge_index = split.train_edge_index.to(device_t)
    val_edge_index = split.val_edge_index.to(device_t)

    y_train = split.train_edge_label.to(device_t).float()
    y_val = split.val_edge_label.to(device_t).float()

    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    pos_w_t = torch.tensor(pos_weight, device=device_t)

    model = GAtrustLike(
        input_dim=features.feature_dim,
        hidden_dim=hidden_dim,
        num_hops=num_hops,
        dropout=dropout,
    ).to(device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc_pr = -1.0
    best_state = None
    patience_t = 0

    for _epoch in range(1, epochs + 1):
        model.train()
        logits = model(x, A_pos, A_neg, train_edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_w_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Early stopping on val AUC-PR
        model.eval()
        with torch.no_grad():
            probs_val, _ent = model.predict_proba(x, A_pos, A_neg, val_edge_index)
            p_val = probs_val[:, 1].numpy()
            auc_pr = float(average_precision_score(y_val.detach().cpu().numpy().astype(np.int64), p_val))

        if auc_pr > best_auc_pr + 1e-8:
            best_auc_pr = auc_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_t = 0
        else:
            patience_t += 1
            if patience_t >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return GATrustBaseline(model=model, x=x, A_pos=A_pos, A_neg=A_neg, device=str(device_t))

