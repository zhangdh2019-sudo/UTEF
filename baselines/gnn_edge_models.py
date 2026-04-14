from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from data.features import FeaturePack
from data.temporal_split import TemporalSignedSplit


def _entropy_from_p(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


class EdgeLogitMLP(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        feat_dim = 4 * hidden_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        u = edge_index[0]
        v = edge_index[1]
        h_u = h[u]
        h_v = h[v]
        feat = torch.cat([h_u, h_v, torch.abs(h_u - h_v), h_u * h_v], dim=1)
        return self.net(feat).squeeze(-1)  # [E]


class SignedGCNLike(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()
        self.W_pos = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.W_neg = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, A_pos: torch.Tensor, A_neg: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(x))
        h = self.dropout(h)
        for l in range(self.num_layers):
            hp = torch.sparse.mm(A_pos, h)
            hn = torch.sparse.mm(A_neg, h)
            h = self.act(self.W_pos[l](hp) + self.W_neg[l](hn))
            h = self.dropout(h)
        return h


class GNNBackbone(nn.Module):
    def __init__(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        heads: int = 2,
    ):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = nn.ReLU()

        if model_type == "mlp":
            self.in_proj = nn.Linear(input_dim, hidden_dim)
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif model_type == "gcn":
            self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        elif model_type == "gat":
            self.convs = nn.ModuleList(
                [GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout)]
                + [GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout) for _ in range(num_layers - 1)]
            )
        elif model_type == "sage":
            self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)] + [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        elif model_type == "signed_gcn":
            # Use signed propagation
            self.signed = SignedGCNLike(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        A_pos: Optional[torch.Tensor] = None,
        A_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model_type == "mlp":
            h = self.act(self.in_proj(x))
            h = self.net(h)
            return h
        if self.model_type == "signed_gcn":
            assert A_pos is not None and A_neg is not None
            return self.signed(x, A_pos, A_neg)
        assert edge_index is not None
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i != len(self.convs) - 1:
                h = self.act(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


@dataclass
class TorchEdgeClassifierBaseline:
    model_type: str
    backbone: GNNBackbone
    head: EdgeLogitMLP
    device: str
    x: torch.Tensor
    edge_index: Optional[torch.Tensor] = None
    A_pos: Optional[torch.Tensor] = None
    A_neg: Optional[torch.Tensor] = None

    def predict_proba(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.backbone.eval()
        self.head.eval()
        with torch.no_grad():
            h = self.backbone(
                self.x,
                edge_index=self.edge_index,
                A_pos=self.A_pos,
                A_neg=self.A_neg,
            )
            logits = self.head(h, edge_index.to(self.x.device))  # [E]
            p = torch.sigmoid(logits)
            ent = _entropy_from_p(p)
            probs = torch.stack([1 - p, p], dim=1)
            return probs.detach().cpu(), ent.detach().cpu()


def _eval_auc_pr_for_torch_model(model: TorchEdgeClassifierBaseline, split: TemporalSignedSplit, device: str) -> float:
    probs, _ = model.predict_proba(split.val_edge_index)
    y = split.val_edge_label.detach().cpu().numpy().astype(np.int64)
    p = probs[:, 1].detach().cpu().numpy()
    return float(average_precision_score(y, p))


def fit_torch_edge_classifier(
    split: TemporalSignedSplit,
    features: FeaturePack,
    model_type: str,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    patience: int = 30,
    device: str = "cpu",
) -> TorchEdgeClassifierBaseline:
    """
    Structure GNN edge classifier baselines (MLP/GCN/GAT/SAGE/SignedGCN).
    """
    device_t = torch.device(device)

    x = features.x.to(device_t)
    y_train = split.train_edge_label.to(device_t).float()
    y_val = split.val_edge_label.to(device_t).float()

    train_edge_index = split.train_edge_index.to(device_t)
    val_edge_index = split.val_edge_index.to(device_t)

    # Positive class weight
    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    pos_w_t = torch.tensor(pos_weight, dtype=torch.float32, device=device_t)

    backbone = GNNBackbone(
        model_type=model_type,
        input_dim=features.feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device_t)
    head = EdgeLogitMLP(hidden_dim=hidden_dim, dropout=dropout).to(device_t)

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)

    best_auc_pr = -1.0
    best_state = None
    patience_t = 0

    # Signed adjacency used only for signed_gcn
    A_pos = split.A_pos.to(device_t) if model_type == "signed_gcn" else None
    A_neg = split.A_neg.to(device_t) if model_type == "signed_gcn" else None

    def predict_auc():
        backbone.eval()
        head.eval()
        with torch.no_grad():
            if model_type == "signed_gcn":
                h = backbone(x, A_pos=A_pos, A_neg=A_neg)
            else:
                h = backbone(x, edge_index=train_edge_index)
            logits = head(h, val_edge_index)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            y = y_val.detach().cpu().numpy().astype(np.int64)
            return float(average_precision_score(y, p))

    for _ in range(epochs):
        backbone.train()
        head.train()

        if model_type == "signed_gcn":
            h = backbone(x, A_pos=A_pos, A_neg=A_neg)
        else:
            h = backbone(x, edge_index=train_edge_index)

        logits = head(h, train_edge_index)  # [E]
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_w_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        auc_pr = predict_auc()
        if auc_pr > best_auc_pr + 1e-8:
            best_auc_pr = auc_pr
            best_state = (deepcopy(backbone.state_dict()), deepcopy(head.state_dict()))
            patience_t = 0
        else:
            patience_t += 1
            if patience_t >= patience:
                break

    if best_state is not None:
        backbone.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])

    return TorchEdgeClassifierBaseline(
        model_type=model_type,
        backbone=backbone,
        head=head,
        device=device,
        x=x.detach(),
        edge_index=train_edge_index.detach(),
        A_pos=A_pos.detach() if A_pos is not None else None,
        A_neg=A_neg.detach() if A_neg is not None else None,
    )

