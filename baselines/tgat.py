from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from data.temporal_split import TemporalSignedSplit


def _predictive_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


class TGATLikeLinkPredictor(nn.Module):
    """
    A practical TGAT-like time-aware neighbor attention baseline.

    Design notes:
      - Neighbor history is built only from train edges (no leakage across splits).
      - For each endpoint, we attend over its K most recent neighbor events from train,
        and mask events that occur after the query timestamp.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        k_history: int = 20,
        time_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.k_history = k_history
        self.time_scale = float(time_scale)

        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.sign_emb = nn.Embedding(2, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

        # Edge classifier
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # These will be registered by `build_history_from_train`
        self.register_buffer("hist_neighbors", torch.full((num_nodes, k_history), -1, dtype=torch.long))
        self.register_buffer("hist_times", torch.zeros((num_nodes, k_history), dtype=torch.long))
        self.register_buffer("hist_signs", torch.zeros((num_nodes, k_history), dtype=torch.long))

    @torch.no_grad()
    def build_history_from_train(self, split: TemporalSignedSplit) -> None:
        src = split.train_edge_index[0].cpu().numpy()
        dst = split.train_edge_index[1].cpu().numpy()
        y = split.train_edge_label.cpu().numpy().astype(np.int64)
        t = split.train_timestamp.cpu().numpy().astype(np.int64)

        events = [[] for _ in range(split.num_nodes)]
        # For each interaction a->b at time ti with sign y, store event for both endpoints.
        for a, b, yi, ti in zip(src, dst, y, t):
            events[a].append((b, ti, yi))
            events[b].append((a, ti, yi))

        for i in range(split.num_nodes):
            if len(events[i]) == 0:
                continue
            # Sort by time, keep last K
            events[i].sort(key=lambda x: x[1])
            tail = events[i][-self.k_history :]
            neigh = [x[0] for x in tail]
            times = [x[1] for x in tail]
            signs = [x[2] for x in tail]
            # Right-align into hist arrays
            pad = self.k_history - len(tail)
            if pad > 0:
                neigh = [-1] * pad + neigh
                times = [0] * pad + times
                signs = [0] * pad + signs
            self.hist_neighbors[i] = torch.tensor(neigh, dtype=torch.long, device=self.hist_neighbors.device)
            self.hist_times[i] = torch.tensor(times, dtype=torch.long, device=self.hist_times.device)
            self.hist_signs[i] = torch.tensor(signs, dtype=torch.long, device=self.hist_signs.device)

    def encode_node(self, node_ids: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
          node_ids: [B]
          query_times: [B] (int64/float)
        Returns:
          rep: [B, hidden_dim]
        """
        B = node_ids.size(0)
        K = self.k_history

        neigh = self.hist_neighbors[node_ids]  # [B,K]
        times = self.hist_times[node_ids]  # [B,K]
        signs = self.hist_signs[node_ids]  # [B,K]

        # Mask: valid event if neighbor != -1 and event_time < query_time
        # Convert to float for delta computations.
        valid = (neigh != -1) & (times < query_times.view(-1, 1))

        # Gather embeddings
        # For invalid neighbors, clamp index to 0 to avoid out-of-range.
        neigh_clamped = neigh.clone()
        neigh_clamped[neigh_clamped == -1] = 0
        neigh_emb = self.node_emb(neigh_clamped)  # [B,K,H]
        sign_emb = self.sign_emb(signs)  # [B,K,H]

        # Time encoding via delta (query_time - event_time)
        delta = (query_times.view(-1, 1).to(torch.float32) - times.to(torch.float32)) / self.time_scale  # [B,K]
        delta = delta.unsqueeze(-1)  # [B,K,1]
        time_emb = self.time_proj(delta)  # [B,K,H]

        keys = neigh_emb + sign_emb + time_emb  # [B,K,H]
        keys = self.dropout(keys)

        query = self.node_emb(node_ids).unsqueeze(1)  # [B,1,H]
        # Dot-product attention
        scores = (query * keys).sum(dim=-1) / (self.hidden_dim**0.5)  # [B,K]
        scores = scores.masked_fill(~valid, -1e9)
        attn = F.softmax(scores, dim=1)  # [B,K]

        rep = (attn.unsqueeze(-1) * keys).sum(dim=1)  # [B,H]
        return rep

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_timestamp: torch.Tensor,
    ) -> torch.Tensor:
        u = edge_index[0]
        v = edge_index[1]
        zu = self.encode_node(u, edge_timestamp)
        zv = self.encode_node(v, edge_timestamp)
        feat = torch.cat([zu, zv, torch.abs(zu - zv), zu * zv], dim=1)
        logits = self.edge_mlp(feat).squeeze(-1)  # [E]
        return logits

    def predict_proba(self, edge_index: torch.Tensor, edge_timestamp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(edge_index, edge_timestamp)
            p = torch.sigmoid(logits)
            ent = _predictive_entropy_from_logits(logits)
            probs = torch.stack([1 - p, p], dim=1)
        return probs.detach().cpu(), ent.detach().cpu()


@dataclass
class TGATBaseline:
    model: TGATLikeLinkPredictor
    device: str

    def predict_proba(self, edge_index: torch.Tensor, edge_timestamp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = edge_index.to(self.device)
        edge_timestamp = edge_timestamp.to(self.device)
        return self.model.predict_proba(edge_index, edge_timestamp)


def fit_tgat_like(
    split: TemporalSignedSplit,
    hidden_dim: int = 64,
    k_history: int = 20,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    patience: int = 30,
    device: str = "cpu",
) -> TGATBaseline:
    device_t = torch.device(device)

    model = TGATLikeLinkPredictor(
        num_nodes=split.num_nodes,
        hidden_dim=hidden_dim,
        k_history=k_history,
        dropout=dropout,
    ).to(device_t)
    model.build_history_from_train(split)
    model = model.to(device_t)

    train_edge_index = split.train_edge_index.to(device_t)
    val_edge_index = split.val_edge_index.to(device_t)
    test_edge_index = split.test_edge_index.to(device_t)

    y_train = split.train_edge_label.to(device_t).float()
    y_val = split.val_edge_label.to(device_t).float()
    y_test = split.test_edge_label.to(device_t).float()

    t_train = split.train_timestamp.to(device_t)
    t_val = split.val_timestamp.to(device_t)

    num_pos = int((y_train == 1).sum().item())
    num_neg = int((y_train == 0).sum().item())
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    pos_w_t = torch.tensor(pos_weight, device=device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc_pr = -1.0
    best_state = None
    patience_t = 0

    for _epoch in range(1, epochs + 1):
        model.train()
        logits = model(train_edge_index, t_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_w_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Val AUC-PR
        model.eval()
        with torch.no_grad():
            logits_val = model(val_edge_index, t_val)
            p_val = torch.sigmoid(logits_val).detach().cpu().numpy()
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

    return TGATBaseline(model=model, device=str(device_t))

