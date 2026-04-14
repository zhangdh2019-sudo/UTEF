from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import reg_loss


class SignedMultiHopPropagation(nn.Module):
    """
    Simple train-only signed propagation with separate positive/negative channels.

    For each hop:
      h_pos = A_pos @ z
      h_neg = A_neg @ z
      z' = act( W_fuse( concat( W_pos h_pos, W_neg h_neg ) ) )
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hops: int,
        dropout: float = 0.0,
        activation: str = "tanh",
    ):
        super().__init__()
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.act = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.W_pos = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.W_neg = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.W_fuse = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_hops)])

    def forward(self, x: torch.Tensor, A_pos: torch.Tensor, A_neg: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [num_nodes, input_dim]
          A_pos/A_neg: sparse [num_nodes, num_nodes]

        Returns:
          z_list stacked into tensor [num_hops+1, num_nodes, hidden_dim]
        """
        z0 = self.act(self.in_proj(x))
        z0 = self.dropout(z0)
        z = z0
        zs = [z0]
        for hop in range(self.num_hops):
            h_pos = torch.sparse.mm(A_pos, z)
            h_neg = torch.sparse.mm(A_neg, z)
            hp_t = self.W_pos[hop](h_pos)
            hn_t = self.W_neg[hop](h_neg)
            z = self.act(self.W_fuse[hop](torch.cat([hp_t, hn_t], dim=-1)))
            z = self.dropout(z)
            zs.append(z)
        return torch.stack(zs, dim=0)


class EdgeEvidenceHead(nn.Module):
    """
    Edge evidence head producing Dirichlet concentration parameters for 2 classes.

    Evidence is constrained to be non-negative via Softplus.
    """

    def __init__(
        self,
        hidden_dim: int,
        evidence_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        extra_dim: int = 5,
    ):
        super().__init__()
        # [z_u, z_v, |z_u-z_v|, z_u*z_v, dot(z_u,z_v), cos(z_u,z_v)]
        feat_base_dim = 4 * hidden_dim + 2
        feat_dim = feat_base_dim + extra_dim
        if evidence_hidden_dim is None:
            # Give the evidential head more capacity for better ranking.
            evidence_hidden_dim = 2 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, evidence_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(evidence_hidden_dim, 2),
        )
        self.softplus = nn.Softplus()

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        extra_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          z: [num_nodes, hidden_dim]
          edge_index: [2, num_edges]
        Returns:
          evidence: [num_edges, 2]
        """
        u = edge_index[0]
        v = edge_index[1]
        z_u = z[u]
        z_v = z[v]
        abs_diff = torch.abs(z_u - z_v)
        prod = z_u * z_v
        dot = (prod).sum(dim=-1, keepdim=True)
        norm_u = torch.norm(z_u, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        norm_v = torch.norm(z_v, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        cos = dot / (norm_u * norm_v)
        edge_feat = torch.cat([z_u, z_v, abs_diff, prod, dot, cos], dim=-1)
        if extra_feat is not None:
            edge_feat = torch.cat([edge_feat, extra_feat], dim=-1)
        evidence = self.softplus(self.mlp(edge_feat))
        return evidence


class TemporalSignedTrustPredictor(nn.Module):
    """
    Temporal Signed Trust Prediction (edge-level evidential) where propagation graph is train-only.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_hops: int = 2,
        kl: float = 0.0,
        dis: float = 0.0,
        u_alpha: float = 0.5,
        dropout: float = 0.0,
        activation: str = "tanh",
        use_temporal_features: bool = True,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.kl = kl
        self.dis = dis
        self.u_alpha = float(u_alpha)
        self.use_temporal_features = bool(use_temporal_features)
        self.hidden_dim = hidden_dim
        # Learnable hop fusion weights for evidential fusion.
        # z_list length is (num_hops + 1) including hop 0.
        self.hop_fusion_logits = nn.Parameter(torch.zeros(num_hops + 1))
        # Simple numeric scaling for temporal features stability.
        self.time_feature_scale = 10.0
        self.propagation = SignedMultiHopPropagation(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            dropout=dropout,
            activation=activation,
        )
        self.edge_head = EdgeEvidenceHead(hidden_dim=hidden_dim, dropout=dropout)

        # temporal stats buffers (set later via build_temporal_stats)
        self.register_buffer("node_last_ts", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("pair_keys_sorted", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("pair_pos_count_sorted", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("pair_neg_count_sorted", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("pair_last_ts_sorted", torch.empty(0, dtype=torch.long), persistent=False)
        self._num_nodes: int = 0

    @torch.no_grad()
    def build_temporal_stats(self, split) -> None:
        """
        Attach train-only temporal stats for temporal encoding.
        """
        self._num_nodes = int(split.num_nodes)
        self.node_last_ts = split.node_last_ts
        self.pair_keys_sorted = split.pair_keys_sorted
        self.pair_pos_count_sorted = split.pair_pos_count_sorted
        self.pair_neg_count_sorted = split.pair_neg_count_sorted
        self.pair_last_ts_sorted = split.pair_last_ts_sorted

    def cal_u(self, alpha: torch.Tensor) -> torch.Tensor:
        # Dirichlet uncertainty: u_dir = K / S where S=sum(alpha) and K=num_classes=2
        K = alpha.size(1)
        S = alpha.sum(dim=1)
        return K / (S + 1e-12)

    def _temporal_edge_features(
        self,
        edge_index: torch.Tensor,
        edge_timestamp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Temporal encoding extra edge features (5 dims):
          [delta_u, delta_v, log1p(pos_count_uv), log1p(neg_count_uv), delta_pair_last]
        where:
          delta_u = log1p(max(0, t - node_last_ts[u])) (0 if no history)
        """
        if self.node_last_ts.numel() == 0:
            # Fallback: return zeros (should not happen if build_temporal_stats called)
            return torch.zeros((edge_index.size(1), 5), dtype=torch.float32, device=edge_timestamp.device)

        u = edge_index[0]
        v = edge_index[1]
        if not self.use_temporal_features:
            return torch.zeros((edge_index.size(1), 5), dtype=torch.float32, device=edge_timestamp.device)

        t = edge_timestamp.to(torch.float32)

        node_last_u = self.node_last_ts[u].to(torch.float32)
        node_last_v = self.node_last_ts[v].to(torch.float32)

        valid_u = node_last_u >= 0
        valid_v = node_last_v >= 0

        delta_u = torch.where(
            valid_u,
            torch.log1p(torch.clamp(t - node_last_u, min=0.0)) / self.time_feature_scale,
            torch.zeros_like(t),
        )
        delta_v = torch.where(
            valid_v,
            torch.log1p(torch.clamp(t - node_last_v, min=0.0)) / self.time_feature_scale,
            torch.zeros_like(t),
        )

        # Pair key lookup
        num_nodes = int(self._num_nodes) if self._num_nodes > 0 else int(self.node_last_ts.numel())
        qkey = u.to(torch.int64) * num_nodes + v.to(torch.int64)  # [E]
        # pair_keys_sorted is sorted; searchsorted for lookup
        idx = torch.searchsorted(self.pair_keys_sorted, qkey)
        valid_idx = idx < self.pair_keys_sorted.numel()
        safe_idx = idx.clone()
        safe_idx[~valid_idx] = 0
        keys_match = self.pair_keys_sorted[safe_idx] == qkey
        valid = valid_idx & keys_match

        pos_count = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_timestamp.device)
        neg_count = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_timestamp.device)
        last_ts = torch.full((edge_index.size(1),), -1, dtype=torch.long, device=edge_timestamp.device)
        if valid.any():
            pos_count[valid] = self.pair_pos_count_sorted[idx[valid]].to(pos_count.device)
            neg_count[valid] = self.pair_neg_count_sorted[idx[valid]].to(neg_count.device)
            last_ts[valid] = self.pair_last_ts_sorted[idx[valid]].to(last_ts.device)

        log1p_pos = torch.log1p(pos_count.to(torch.float32))
        log1p_neg = torch.log1p(neg_count.to(torch.float32))

        valid_last = last_ts >= 0
        delta_pair_last = torch.where(
            valid_last,
            torch.log1p(torch.clamp(t - last_ts.to(torch.float32), min=0.0)) / self.time_feature_scale,
            torch.zeros_like(t),
        )

        extra = torch.stack([delta_u, delta_v, log1p_pos, log1p_neg, delta_pair_last], dim=1)
        return extra

    def forward(
        self,
        x: torch.Tensor,
        A_pos: torch.Tensor,
        A_neg: torch.Tensor,
        edge_index: torch.Tensor,
        edge_timestamp: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        If y is provided, compute loss (classification aligned with BCE + evidential regularization).
        """
        # z_list: [num_hops+1, num_nodes, hidden_dim]
        z_list = self.propagation(x, A_pos, A_neg)

        extra_feat = self._temporal_edge_features(edge_index=edge_index, edge_timestamp=edge_timestamp)

        # Evidential fusion: learn attention weights (multi-hop evidence != independent evidence).
        evidence_list = []
        for hop in range(z_list.size(0)):
            evidence_h = self.edge_head(z_list[hop], edge_index, extra_feat=extra_feat)  # [E,2]
            evidence_list.append(evidence_h)
        evidence_stack = torch.stack(evidence_list, dim=0)  # [H, E, 2]
        hop_weights = F.softmax(self.hop_fusion_logits, dim=0).view(-1, 1, 1)  # [H,1,1]
        evidence_a = (hop_weights * evidence_stack).sum(dim=0)  # [E,2]

        alpha_a = evidence_a + 1.0
        probs = alpha_a / alpha_a.sum(dim=1, keepdim=True)  # [E,2]

        u_dir = self.cal_u(alpha_a)  # [E]
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)  # [E]
        # Blend Dirichlet uncertainty with entropy.
        u_a = self.u_alpha * u_dir + (1.0 - self.u_alpha) * entropy

        # trust logit for optional ranking losses / metrics
        eps = 1e-12
        trust_logit = torch.log(probs[:, 1] + eps) - torch.log(probs[:, 0] + eps)

        out = {
            "evidence": evidence_a,
            "alpha": alpha_a,
            "u": u_a,
            "probs": probs,
            "trust_logit": trust_logit,
        }

        if y is not None:
            p_trust = probs[:, 1].clamp(1e-12, 1 - 1e-12)
            bce = F.binary_cross_entropy(p_trust, y.float(), reduction="none").mean()
            uncertainty_reg = reg_loss(y, evidence_a, c=2, kl=self.kl, dis=self.dis).mean()
            out["loss"] = bce + uncertainty_reg
        return out

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        A_pos: torch.Tensor,
        A_neg: torch.Tensor,
        edge_index: torch.Tensor,
        edge_timestamp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, A_pos, A_neg, edge_index, edge_timestamp=edge_timestamp, y=None)
        # probs: [E,2]
        return out["probs"], out["u"]

