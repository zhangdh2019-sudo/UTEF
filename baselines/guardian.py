from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from data.features import FeaturePack
from data.temporal_split import TemporalSignedSplit


def _edge_feat(x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    u = edge_index[0]
    v = edge_index[1]
    xu = x[u]
    xv = x[v]
    dot = (xu * xv).sum(dim=1, keepdim=True)
    feat = torch.cat([xu, xv, torch.abs(xu - xv), xu * xv, dot], dim=1)
    return feat.detach().cpu().numpy()


@dataclass
class GuardianBaseline:
    clf: LogisticRegression
    x: torch.Tensor

    def predict_proba(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X = _edge_feat(self.x, edge_index)
        p = self.clf.predict_proba(X)[:, 1]
        p_t = torch.tensor(p, dtype=torch.float32)
        # Guardian-like confidence gate proxy: entropy as uncertainty
        u = -(p_t * torch.log(p_t + 1e-12) + (1 - p_t) * torch.log(1 - p_t + 1e-12))
        probs = torch.stack([1 - p_t, p_t], dim=1)
        return probs, u


def fit_guardian(split: TemporalSignedSplit, features: FeaturePack, device: str = "cpu") -> GuardianBaseline:
    """
    Guardian-style shallow edge classifier under the same protocol/features as other models.
    No handcrafted feature override is allowed here.
    """
    _ = device
    x = features.x.cpu()
    X_train = _edge_feat(x, split.train_edge_index)
    y = split.train_edge_label.cpu().numpy().astype(np.int64)

    num_pos = int((y == 1).sum())
    num_neg = int((y == 0).sum())
    w_pos = (num_pos + num_neg) / (2.0 * max(num_pos, 1))
    w_neg = (num_pos + num_neg) / (2.0 * max(num_neg, 1))
    clf = LogisticRegression(max_iter=2500, solver="liblinear", class_weight={0: w_neg, 1: w_pos})
    clf.fit(X_train, y)
    return GuardianBaseline(clf=clf, x=x)

