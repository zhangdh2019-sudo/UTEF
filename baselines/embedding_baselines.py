from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from data.temporal_split import TemporalSignedSplit


def _edge_feature_from_embeddings(z: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    u = edge_index[0]
    v = edge_index[1]
    z_u = z[u]
    z_v = z[v]
    feat = torch.cat([z_u, z_v, torch.abs(z_u - z_v), z_u * z_v], dim=1)
    return feat.detach().cpu().numpy()


def _predictive_entropy(p_trust: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p_trust, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@dataclass
class EmbeddingBasedEdgeClassifier:
    lr: LogisticRegression
    edge_emb_mode: str  # "pos_only", "signed_concat", etc.
    z_pos: torch.Tensor
    z_neg: Optional[torch.Tensor] = None
    device: str = "cpu"

    def _get_edge_embeddings(self, edge_index: torch.Tensor) -> np.ndarray:
        if self.edge_emb_mode == "pos_only":
            X = _edge_feature_from_embeddings(self.z_pos, edge_index)
            return X

        if self.edge_emb_mode == "signed_concat":
            if self.z_neg is None:
                raise ValueError("z_neg is required for signed_concat")
            z = torch.cat([self.z_pos, self.z_neg], dim=1)
            X = _edge_feature_from_embeddings(z, edge_index)
            return X

        raise ValueError(f"Unknown edge_emb_mode: {self.edge_emb_mode}")

    def predict_proba(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_feat = self._get_edge_embeddings(edge_index)
        p = self.lr.predict_proba(edge_feat)[:, 1]
        ent = _predictive_entropy(p)
        probs = np.stack([1 - p, p], axis=1).astype(np.float32)
        return torch.tensor(probs), torch.tensor(ent, dtype=torch.float32)


def _train_node2vec_embeddings(
    edge_index: torch.Tensor,
    num_nodes: int,
    embedding_dim: int = 64,
    walk_length: int = 20,
    context_size: int = 10,
    epochs: int = 30,
    lr: float = 0.01,
    device: str = "cpu",
) -> torch.Tensor:
    from torch_geometric.nn import Node2Vec

    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, embedding_dim, device=device)

    # Undirected for random walks
    edge_index_und = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index_und = torch.unique(edge_index_und, dim=1)

    model = Node2Vec(
        edge_index_und.to(device),
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        num_negative_samples=1,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    model.train()
    for _ in range(epochs):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

    return model.embedding.weight.detach().to(device)


def fit_node2vec_edge_classifier(
    split: TemporalSignedSplit,
    embedding_dim: int = 64,
    device: str = "cpu",
) -> EmbeddingBasedEdgeClassifier:
    """
    Node2Vec baseline (pos-only): learn node embeddings on positive edges,
    then train LogisticRegression on edge pairs.
    """
    pos_mask = split.train_edge_label == 1
    edge_index_pos = split.train_edge_index[:, pos_mask]
    z_pos = _train_node2vec_embeddings(
        edge_index=edge_index_pos,
        num_nodes=split.num_nodes,
        embedding_dim=embedding_dim,
        device=device,
    )

    edge_feat = _edge_feature_from_embeddings(z_pos, split.train_edge_index)
    y = split.train_edge_label.detach().cpu().numpy().astype(np.int64)

    num_pos = int((y == 1).sum())
    num_neg = int((y == 0).sum())
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    class_weight = {0: 1.0, 1: pos_weight}

    lr = LogisticRegression(max_iter=2000, solver="liblinear", class_weight=class_weight)
    lr.fit(edge_feat, y)

    return EmbeddingBasedEdgeClassifier(lr=lr, edge_emb_mode="pos_only", z_pos=z_pos, z_neg=None, device=device)


def fit_signed_node_embedding_edge_classifier(
    split: TemporalSignedSplit,
    embedding_dim: int = 64,
    device: str = "cpu",
    signed_mode: str = "signed_concat",
) -> EmbeddingBasedEdgeClassifier:
    """
    SiNE / SIDE style approximation:
      - Train separate Node2Vec embeddings on positive and negative subgraphs.
      - Combine node embeddings into a signed embedding and train an edge classifier.

    This is a practical signed-embedding baseline when exact SiNE/SIDE code is unavailable.
    """
    pos_mask = split.train_edge_label == 1
    neg_mask = split.train_edge_label == 0

    edge_index_pos = split.train_edge_index[:, pos_mask]
    edge_index_neg = split.train_edge_index[:, neg_mask]

    z_pos = _train_node2vec_embeddings(
        edge_index=edge_index_pos,
        num_nodes=split.num_nodes,
        embedding_dim=embedding_dim,
        device=device,
    )
    z_neg = _train_node2vec_embeddings(
        edge_index=edge_index_neg,
        num_nodes=split.num_nodes,
        embedding_dim=embedding_dim,
        device=device,
    )

    edge_feat = _edge_feature_from_embeddings(torch.cat([z_pos, z_neg], dim=1), split.train_edge_index)
    y = split.train_edge_label.detach().cpu().numpy().astype(np.int64)

    num_pos = int((y == 1).sum())
    num_neg = int((y == 0).sum())
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    class_weight = {0: 1.0, 1: pos_weight}

    lr = LogisticRegression(max_iter=2000, solver="liblinear", class_weight=class_weight)
    lr.fit(edge_feat, y)

    return EmbeddingBasedEdgeClassifier(
        lr=lr,
        edge_emb_mode="signed_concat",
        z_pos=z_pos,
        z_neg=z_neg,
        device=device,
    )


@dataclass
class SDNELikeEmbeddingModel:
    z: torch.Tensor  # [num_nodes, dim]
    dim: int
    device: str

    def predict_proba(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = edge_index[0]
        v = edge_index[1]
        logits = (self.z[u] * self.z[v]).sum(dim=1)
        p = torch.sigmoid(logits)
        ent = -(p * torch.log(p + 1e-12) + (1 - p) * torch.log(1 - p + 1e-12))
        probs = torch.stack([1 - p, p], dim=1)
        return probs.detach().cpu(), ent.detach().cpu()


def fit_sdne_like_embedding(
    split: TemporalSignedSplit,
    dim: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    device: str = "cpu",
) -> SDNELikeEmbeddingModel:
    """
    SDNE-style structural embedding baseline (practical variant):
      learn node embeddings z with an inner-product decoder using supervised sign labels.
    """
    z = torch.randn(split.num_nodes, dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    # Pos weight for BCE to match signed imbalance.
    y = split.train_edge_label.to(device).float()
    num_pos = int((y == 1).sum().item())
    num_neg = int((y == 0).sum().item())
    pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0

    edge_index = split.train_edge_index.to(device)
    u = edge_index[0]
    v = edge_index[1]

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = (z[u] * z[v]).sum(dim=1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(pos_weight, device=device))
        loss.backward()
        optimizer.step()

    return SDNELikeEmbeddingModel(z=z.detach(), dim=dim, device=device)

