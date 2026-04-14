from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from data.features import build_node_features
from data.temporal_split import TemporalSignedSplit


def _edge_features_from_node_features(x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    """
    Build edge features from node features via:
      [x_u, x_v, x_u-x_v, x_u*x_v]
    """
    u = edge_index[0]
    v = edge_index[1]
    x_u = x[u]
    x_v = x[v]
    edge_feat = torch.cat([x_u, x_v, x_u - x_v, x_u * x_v], dim=1)
    return edge_feat.detach().cpu().numpy()


def _predictive_entropy(p_trust: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p_trust, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@dataclass
class HeuristicTrustModel:
    lr: LogisticRegression
    node_x: torch.Tensor

    def predict_proba(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_feat = _edge_features_from_node_features(self.node_x, edge_index)
        p = self.lr.predict_proba(edge_feat)[:, 1]  # trust probability
        ent = _predictive_entropy(p)
        return torch.tensor(np.stack([1 - p, p], axis=1), dtype=torch.float32), torch.tensor(ent, dtype=torch.float32)


def fit_heuristic_trust(
    split: TemporalSignedSplit,
    pos_weight: Optional[float] = None,
    device: str = "cpu",
) -> HeuristicTrustModel:
    """
    HeuristicTrust (Balance/Status + LogisticRegression) baseline.

    We always use `signed_structural` node features to expose balance/status proxies.
    """
    # Build balance/status node features on train-visible graph only.
    feat = build_node_features(split, feature_type="signed_structural", device=device)
    node_x = feat.x.to(torch.device(device))

    # Training edge features
    X_train = _edge_features_from_node_features(node_x, split.train_edge_index)
    y_train = split.train_edge_label.detach().cpu().numpy().astype(np.int64)

    if pos_weight is None:
        num_pos = int((y_train == 1).sum())
        num_neg = int((y_train == 0).sum())
        if num_pos == 0:
            pos_weight = 1.0
        else:
            pos_weight = float(num_neg / num_pos)

    class_weight = {0: 1.0, 1: pos_weight}
    lr = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight=class_weight,
    )
    lr.fit(X_train, y_train)
    return HeuristicTrustModel(lr=lr, node_x=node_x)

