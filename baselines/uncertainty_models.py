from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from utils import ce_loss, reg_loss


def predictive_entropy_binary(p_trust: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p_trust.clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


@dataclass
class MCDropoutResult:
    probs: torch.Tensor  # [E,2]
    uncertainty: torch.Tensor  # [E]


@torch.no_grad()
def predict_mc_dropout_dirichlet(
    model,
    x: torch.Tensor,
    A_pos: torch.Tensor,
    A_neg: torch.Tensor,
    edge_index: torch.Tensor,
    edge_timestamp: Optional[torch.Tensor] = None,
    mc_samples: int = 20,
) -> MCDropoutResult:
    """
    MC Dropout for evidential Dirichlet model:
      - sample multiple forward passes with dropout enabled
      - average probabilities
      - uncertainty = predictive entropy of the averaged probabilities
    """
    model.eval()
    # Enable dropout layers
    def _enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()

    model.apply(_enable_dropout)

    probs_samples = []
    u_samples = []
    for _ in range(mc_samples):
        if edge_timestamp is None:
            out = model(x, A_pos, A_neg, edge_index, y=None)
        else:
            out = model(x, A_pos, A_neg, edge_index, edge_timestamp=edge_timestamp, y=None)
        probs = out["probs"]  # [E,2]
        probs_samples.append(probs.unsqueeze(0))
        u_samples.append(out["u"].unsqueeze(0))  # [E,1]

    probs_mean = torch.mean(torch.cat(probs_samples, dim=0), dim=0)
    p_trust = probs_mean[:, 1]
    ent = predictive_entropy_binary(p_trust)
    return MCDropoutResult(probs=probs_mean.detach().cpu(), uncertainty=ent.detach().cpu())


@dataclass
class EnsembleResult:
    probs: torch.Tensor
    uncertainty: torch.Tensor


@torch.no_grad()
def predict_ensemble_mean(
    models: List,
    x: torch.Tensor,
    A_pos: torch.Tensor,
    A_neg: torch.Tensor,
    edge_index: torch.Tensor,
) -> EnsembleResult:
    probs_list = []
    for m in models:
        out = m(x, A_pos, A_neg, edge_index, y=None)
        probs_list.append(out["probs"].unsqueeze(0))
    probs_stack = torch.cat(probs_list, dim=0)  # [M,E,2]
    probs_mean = probs_stack.mean(dim=0)
    p_trust = probs_mean[:, 1]
    ent = predictive_entropy_binary(p_trust)
    # Optionally: epistemic uncertainty via variance of probs
    # var = probs_stack[:, :, 1].var(dim=0)
    return EnsembleResult(probs=probs_mean.detach().cpu(), uncertainty=ent.detach().cpu())


class TemperatureScalingBinary(nn.Module):
    """
    Temperature scaling wrapper for binary logits.
    """

    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor(init_T, dtype=torch.float32)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T)
        return logits / T

    @torch.no_grad()
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        logits_t = self.forward(logits)
        p = torch.sigmoid(logits_t)
        return torch.stack([1 - p, p], dim=1)


def _fit_temperature_scaling_binary(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.01,
) -> TemperatureScalingBinary:
    scaler = TemperatureScalingBinary(init_T=1.0).to(logits.device)
    optimizer = torch.optim.LBFGS([scaler.log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        logits_t = scaler(logits)
        p = torch.sigmoid(logits_t).clamp(1e-12, 1 - 1e-12)
        nll = -(y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p)).mean()
        nll.backward()
        return nll

    optimizer.step(closure)
    return scaler


@dataclass
class EvidentialMLPModel:
    mlp: nn.Module
    device: str

    def predict_proba(self, edge_index: torch.Tensor, node_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.mlp.eval()
        with torch.no_grad():
            probs, u = self.mlp.predict_proba(edge_index, node_x)
        return probs.detach().cpu(), u.detach().cpu()


class EvidentialMLPEdge(nn.Module):
    """
    Evidential Dirichlet MLP on edges without graph propagation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kl: float = 0.0,
        dis: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kl = kl
        self.dis = dis
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.evidence_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.softplus = nn.Softplus()

    def forward(self, edge_index: torch.Tensor, node_x: torch.Tensor, y: Optional[torch.Tensor] = None) -> dict:
        h = torch.tanh(self.proj(node_x))
        h = self.dropout(h)
        u = edge_index[0]
        v = edge_index[1]
        z_u = h[u]
        z_v = h[v]
        feat = torch.cat([z_u, z_v, torch.abs(z_u - z_v), z_u * z_v], dim=1)
        evidence = self.softplus(self.evidence_mlp(feat))  # [E,2]
        alpha = evidence + 1.0
        K = alpha.size(1)
        S = alpha.sum(dim=1, keepdim=True)
        uncertainty = (K / (S + 1e-12)).squeeze(-1)
        probs = alpha / alpha.sum(dim=1, keepdim=True)

        out = {"evidence": evidence, "alpha": alpha, "u": uncertainty, "probs": probs}
        if y is not None:
            classification = ce_loss(y, alpha, c=2).mean()
            uncertainty_reg = reg_loss(y, evidence, c=2, kl=self.kl, dis=self.dis).mean()
            out["loss"] = classification + uncertainty_reg
        return out

    @torch.no_grad()
    def predict_proba(self, edge_index: torch.Tensor, node_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(edge_index, node_x, y=None)
        return out["probs"], out["u"]


def fit_evidential_mlp(
    split,
    features,
    hidden_dim: int = 64,
    kl: float = 0.0,
    dis: float = 0.0,
    dropout: float = 0.1,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
) -> EvidentialMLPModel:
    device_t = torch.device(device)
    node_x = features.x.to(device_t)

    y_train = split.train_edge_label.to(device_t)
    y_val = split.val_edge_label.to(device_t)

    train_edge_index = split.train_edge_index.to(device_t)
    val_edge_index = split.val_edge_index.to(device_t)

    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    pos_w_t = torch.tensor(pos_weight, device=device_t, dtype=torch.float32)
    edge_weights = torch.where(y_train == 1, pos_w_t, torch.tensor(1.0, device=device_t))

    model = EvidentialMLPEdge(
        input_dim=features.feature_dim,
        hidden_dim=hidden_dim,
        kl=kl,
        dis=dis,
        dropout=dropout,
    ).to(device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_auc_pr = -1.0
    best_state = None
    patience_t = 0

    for _epoch in range(1, epochs + 1):
        model.train()
        out = model(train_edge_index, node_x, y=None)
        alpha = out["alpha"]
        evidence = out["evidence"]

        classification = ce_loss(y_train, alpha, c=2).squeeze(-1)
        classification_loss = (classification * edge_weights).mean()
        uncertainty_reg = reg_loss(y_train, evidence, c=2, kl=kl, dis=dis).squeeze(-1).mean()
        loss = classification_loss + uncertainty_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_edge_index, node_x, y=None)
            p_val = val_out["probs"][:, 1].detach().cpu().numpy()
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
    return EvidentialMLPModel(mlp=model, device=str(device_t))

