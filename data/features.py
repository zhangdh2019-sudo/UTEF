import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .temporal_split import TemporalSignedSplit


@dataclass
class FeaturePack:
    x: torch.Tensor  # [num_nodes, feat_dim]
    feature_type: str
    feature_names: Tuple[str, ...]

    @property
    def feature_dim(self) -> int:
        return int(self.x.size(1))


def _learn_edge2vec_embeddings(
    edge_index_pos: torch.Tensor,
    num_nodes: int,
    emb_dim: int,
    epochs: int,
    lr: float,
    device: torch.device,
    seed: Optional[int],
) -> torch.Tensor:
    """
    Pure PyTorch fallback embedding learner (negative sampling + dot-product BCE).
    This keeps the "no handcrafted features" requirement when pyg Node2Vec deps are missing.
    """
    if seed is not None:
        s = int(seed)
        torch.manual_seed(s)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(s)

    emb = torch.nn.Embedding(num_nodes, emb_dim).to(device)
    torch.nn.init.xavier_uniform_(emb.weight)
    optimizer = torch.optim.Adam(emb.parameters(), lr=lr)

    src_all = edge_index_pos[0].to(device)
    dst_all = edge_index_pos[1].to(device)
    n_pos = src_all.numel()
    if n_pos == 0:
        return torch.zeros(num_nodes, emb_dim, device=device)

    batch_size = min(4096, max(512, n_pos))
    steps_per_epoch = max(20, n_pos // max(1, batch_size))

    for _ in range(max(20, epochs * 2)):
        emb.train()
        for _ in range(steps_per_epoch):
            pos_idx = torch.randint(0, n_pos, (batch_size,), device=device)
            u = src_all[pos_idx]
            v_pos = dst_all[pos_idx]
            v_neg = torch.randint(0, num_nodes, (batch_size,), device=device)

            eu = emb(u)
            evp = emb(v_pos)
            evn = emb(v_neg)

            pos_logits = (eu * evp).sum(dim=1)
            neg_logits = (eu * evn).sum(dim=1)

            pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )
            neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return emb.weight.detach()


def _pagerank_pytorch(edge_index: torch.Tensor, num_nodes: int, damping: float = 0.85, iters: int = 30) -> torch.Tensor:
    """
    Simple PageRank via power iteration on a directed graph.

    edge_index is [2, E] with u->v meaning outgoing from u.
    """
    device = edge_index.device
    u = edge_index[0]
    v = edge_index[1]
    E = u.numel()
    if E == 0:
        return torch.ones(num_nodes, device=device) / num_nodes

    out_deg = torch.bincount(u, minlength=num_nodes).float().to(device)
    out_deg = out_deg.clamp_min(1.0)
    values = 1.0 / out_deg[u]

    # Build A_norm where A[u,v] = 1/out_deg[u]
    idx = torch.stack([u, v], dim=0)
    A = torch.sparse_coo_tensor(idx, values, (num_nodes, num_nodes)).coalesce()

    # Power iteration: p_{k+1} = damping * A^T p_k + (1-d)/N
    p = torch.ones(num_nodes, device=device) / num_nodes
    one_minus = (1.0 - damping) / num_nodes

    # Compute A^T @ p without explicit transpose by swapping indices.
    idx_t = torch.stack([v, u], dim=0)
    A_t = torch.sparse_coo_tensor(idx_t, values, (num_nodes, num_nodes)).coalesce()

    for _ in range(iters):
        p = damping * torch.sparse.mm(A_t, p.view(-1, 1)).view(-1) + one_minus
    return p


def _compute_degrees(split: TemporalSignedSplit) -> Dict[str, torch.Tensor]:
    src, dst = split.train_edge_index[0], split.train_edge_index[1]
    num_nodes = split.num_nodes
    labels = split.train_edge_label

    out_deg = torch.bincount(src, minlength=num_nodes).float()
    in_deg = torch.bincount(dst, minlength=num_nodes).float()
    total_deg = out_deg + in_deg

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_out = torch.bincount(src[pos_mask], minlength=num_nodes).float()
    pos_in = torch.bincount(dst[pos_mask], minlength=num_nodes).float()
    neg_out = torch.bincount(src[neg_mask], minlength=num_nodes).float()
    neg_in = torch.bincount(dst[neg_mask], minlength=num_nodes).float()

    eps = 1e-12
    pos_ratio = pos_out.add(pos_in) / (pos_out.add(pos_in) + neg_out.add(neg_in) + eps)
    neg_ratio = 1.0 - pos_ratio

    # Signed balance based on counts (positive minus negative over total)
    signed_balance = (pos_out.add(pos_in) - neg_out.add(neg_in)) / (pos_out.add(pos_in) + neg_out.add(neg_in) + eps)

    return {
        "in_deg": in_deg,
        "out_deg": out_deg,
        "total_deg": total_deg,
        "pos_in_deg": pos_in,
        "neg_in_deg": neg_in,
        "pos_out_deg": pos_out,
        "neg_out_deg": neg_out,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
        "signed_balance": signed_balance,
    }


def _reciprocity_pos(split: TemporalSignedSplit) -> torch.Tensor:
    """
    Node-level reciprocity for positive edges:
      reciprocity_pos[u] = (# of positive outgoing edges u->v such that v->u is positive) / (pos_out_deg[u] + eps)
    """
    src, dst = split.train_edge_index[0], split.train_edge_index[1]
    labels = split.train_edge_label
    num_nodes = split.num_nodes

    pos_mask = labels == 1
    src_pos = src[pos_mask]
    dst_pos = dst[pos_mask]

    eps = 1e-12
    pos_out_deg = torch.bincount(src_pos, minlength=num_nodes).float()

    # Build a quick set of positive directed edges for membership queries.
    # For N ~ 6k, E ~ 32k this is fine.
    edges_pos = src_pos.to(torch.int64) * num_nodes + dst_pos.to(torch.int64)
    # Use unique membership via sort+search trick.
    edges_pos_sorted, _ = torch.sort(edges_pos)

    # For each positive edge (u->v), check if (v->u) exists (positive).
    check_key = dst_pos.to(torch.int64) * num_nodes + src_pos.to(torch.int64)
    # binary search membership
    idx = torch.searchsorted(edges_pos_sorted, check_key)
    mask_valid = idx < edges_pos_sorted.numel()
    exists = torch.zeros_like(mask_valid, dtype=torch.bool)
    if mask_valid.any():
        exists[mask_valid] = edges_pos_sorted[idx[mask_valid]] == check_key[mask_valid]

    # For each u, count reciprocal positive edges among its positive out-edges.
    recip_count = torch.zeros(num_nodes, dtype=torch.float32)
    recip_count.scatter_add_(0, src_pos, exists.float().cpu())
    recip_count = recip_count.to(src.device)
    reciprocity_pos = recip_count / (pos_out_deg + eps)
    return reciprocity_pos


def _reciprocity_pos_to_neg(split: TemporalSignedSplit) -> torch.Tensor:
    """
    Cross reciprocity ratio:
      pos_recip_neg[u] = (# pos out edges u->v where v->u is negative) / (pos_out_deg[u] + eps)
    """
    src, dst = split.train_edge_index[0], split.train_edge_index[1]
    labels = split.train_edge_label
    num_nodes = split.num_nodes
    pos_mask = labels == 1

    src_pos = src[pos_mask]
    dst_pos = dst[pos_mask]
    eps = 1e-12
    pos_out_deg = torch.bincount(src_pos, minlength=num_nodes).float()

    edges_neg = (src[labels == 0].to(torch.int64) * num_nodes + dst[labels == 0].to(torch.int64)).cpu()
    if edges_neg.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.float32, device=src.device)

    edges_neg_sorted, _ = torch.sort(edges_neg)
    check_key = dst_pos.to(torch.int64) * num_nodes + src_pos.to(torch.int64)
    idx = torch.searchsorted(edges_neg_sorted, check_key)
    mask_valid = idx < edges_neg_sorted.numel()
    exists = torch.zeros_like(mask_valid, dtype=torch.bool)
    if mask_valid.any():
        exists[mask_valid] = edges_neg_sorted[idx[mask_valid]] == check_key[mask_valid].cpu()

    recip_count = torch.zeros(num_nodes, dtype=torch.float32)
    recip_count.scatter_add_(0, src_pos, exists.float())
    recip_count = recip_count.to(src.device)
    pos_recip_neg = recip_count / (pos_out_deg + eps)
    return pos_recip_neg


def _reciprocity_neg_to_pos(split: TemporalSignedSplit) -> torch.Tensor:
    """
    Cross reciprocity ratio:
      neg_recip_pos[u] = (# neg out edges u->v where v->u is positive) / (neg_out_deg[u] + eps)
    """
    src, dst = split.train_edge_index[0], split.train_edge_index[1]
    labels = split.train_edge_label
    num_nodes = split.num_nodes
    neg_mask = labels == 0

    src_neg = src[neg_mask]
    dst_neg = dst[neg_mask]
    eps = 1e-12
    neg_out_deg = torch.bincount(src_neg, minlength=num_nodes).float()

    edges_pos = (src[labels == 1].to(torch.int64) * num_nodes + dst[labels == 1].to(torch.int64)).cpu()
    if edges_pos.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.float32, device=src.device)

    edges_pos_sorted, _ = torch.sort(edges_pos)
    check_key = dst_neg.to(torch.int64) * num_nodes + src_neg.to(torch.int64)
    idx = torch.searchsorted(edges_pos_sorted, check_key)
    mask_valid = idx < edges_pos_sorted.numel()
    exists = torch.zeros_like(mask_valid, dtype=torch.bool)
    if mask_valid.any():
        exists[mask_valid] = edges_pos_sorted[idx[mask_valid]] == check_key[mask_valid].cpu()

    recip_count = torch.zeros(num_nodes, dtype=torch.float32)
    recip_count.scatter_add_(0, src_neg, exists.float())
    recip_count = recip_count.to(src.device)
    neg_recip_pos = recip_count / (neg_out_deg + eps)
    return neg_recip_pos


def build_node_features(
    split: TemporalSignedSplit,
    feature_type: str,
    node2vec_dim: int = 64,
    node2vec_walk_length: int = 20,
    node2vec_context_size: int = 10,
    node2vec_steps: int = 1000,
    node2vec_epochs: int = 50,
    node2vec_lr: float = 0.01,
    device: str = "cpu",
    cache_dir: str = "data/processed",
    node2vec_random_seed: Optional[int] = None,
) -> FeaturePack:
    """
    feature_type:
      - minimal
      - signed_structural
      - node2vec
    """
    feature_type = feature_type.lower()
    device_t = torch.device(device)
    os.makedirs(cache_dir, exist_ok=True)

    if feature_type == "minimal":
        src, dst = split.train_edge_index[0].to(device_t), split.train_edge_index[1].to(device_t)
        degs = _compute_degrees(split)
        pagerank = _pagerank_pytorch(split.train_edge_index.to(device_t), split.num_nodes).to(device_t)

        x = torch.stack(
            [
                degs["in_deg"].to(device_t),
                degs["out_deg"].to(device_t),
                degs["total_deg"].to(device_t),
                pagerank,
            ],
            dim=1,
        )
        names = ("in_degree", "out_degree", "total_degree", "pagerank")
        return FeaturePack(x=x, feature_type=feature_type, feature_names=names)

    if feature_type == "signed_structural":
        degs = _compute_degrees(split)
        pagerank = _pagerank_pytorch(split.train_edge_index.to(device_t), split.num_nodes).to(device_t)
        recip_pos = _reciprocity_pos(split).to(device_t)
        recip_pos_neg = _reciprocity_pos_to_neg(split).to(device_t)
        recip_neg_pos = _reciprocity_neg_to_pos(split).to(device_t)

        minimal_x = torch.stack(
            [
                degs["in_deg"].to(device_t),
                degs["out_deg"].to(device_t),
                degs["total_deg"].to(device_t),
                pagerank,
            ],
            dim=1,
        )

        extra_x = torch.stack(
            [
                degs["pos_in_deg"].to(device_t),
                degs["neg_in_deg"].to(device_t),
                degs["pos_out_deg"].to(device_t),
                degs["neg_out_deg"].to(device_t),
                degs["pos_ratio"].to(device_t),
                degs["neg_ratio"].to(device_t),
                degs["signed_balance"].to(device_t),
                recip_pos,
                recip_pos_neg,
                recip_neg_pos,
            ],
            dim=1,
        )
        x = torch.cat([minimal_x, extra_x], dim=1)
        names = (
            "in_degree",
            "out_degree",
            "total_degree",
            "pagerank",
            "pos_in_degree",
            "neg_in_degree",
            "pos_out_degree",
            "neg_out_degree",
            "pos_ratio",
            "neg_ratio",
            "signed_balance",
            "reciprocity_pos",
            "pos_recip_neg_ratio",
            "neg_recip_pos_ratio",
        )
        return FeaturePack(x=x, feature_type=feature_type, feature_names=names)

    if feature_type == "node2vec":
        # Node2Vec features should be isolated: no fallback to structural features.
        # We train node2vec on positive edges from the train split.
        pos_mask = split.train_edge_label == 1
        edge_index_pos = split.train_edge_index[:, pos_mask]
        if edge_index_pos.numel() == 0:
            x = torch.zeros(split.num_nodes, node2vec_dim, device=device_t)
            return FeaturePack(x=x, feature_type=feature_type, feature_names=tuple([f"n2v_{i}" for i in range(node2vec_dim)]))

        # Make undirected for random walks by adding reverse edges.
        edge_index_und = torch.cat([edge_index_pos, edge_index_pos.flip(0)], dim=1)
        edge_index_und = torch.unique(edge_index_und, dim=1)

        seed_part = "none" if node2vec_random_seed is None else str(int(node2vec_random_seed))
        dataset_tag = getattr(split, "dataset_name", "dataset")
        cache_tag = (
            f"{dataset_tag}_node2vec_dim{node2vec_dim}_walk{node2vec_walk_length}"
            f"_ctx{node2vec_context_size}_steps{node2vec_steps}_ep{node2vec_epochs}"
            f"_lr{node2vec_lr}_seed{seed_part}"
        )
        cache_path = os.path.join(cache_dir, cache_tag + ".pt")
        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path, map_location=device_t, weights_only=False)
            except TypeError:
                cached = torch.load(cache_path, map_location=device_t)
            x = cached["x"].to(device_t)
            return FeaturePack(x=x, feature_type=feature_type, feature_names=tuple([f"n2v_{i}" for i in range(node2vec_dim)]))

        x = None
        try:
            from torch_geometric.nn import Node2Vec

            if node2vec_random_seed is not None:
                s = int(node2vec_random_seed)
                torch.manual_seed(s)
                if device_t.type == "cuda":
                    torch.cuda.manual_seed_all(s)

            model = Node2Vec(
                edge_index_und.to(device_t),
                embedding_dim=node2vec_dim,
                walk_length=node2vec_walk_length,
                context_size=node2vec_context_size,
                num_negative_samples=1,
                sparse=True,
            ).to(device_t)

            loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
            optimizer = torch.optim.Adam(list(model.parameters()), lr=node2vec_lr)

            model.train()
            for _ in range(node2vec_epochs):
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()
            x = model.embedding.weight.detach().to(device_t)
        except Exception:
            x = _learn_edge2vec_embeddings(
                edge_index_pos=edge_index_pos,
                num_nodes=split.num_nodes,
                emb_dim=node2vec_dim,
                epochs=node2vec_epochs,
                lr=node2vec_lr,
                device=device_t,
                seed=node2vec_random_seed,
            ).to(device_t)

        torch.save({"x": x.cpu()}, cache_path)
        return FeaturePack(x=x, feature_type=feature_type, feature_names=tuple([f"n2v_{i}" for i in range(node2vec_dim)]))

    raise ValueError(f"Unknown feature_type: {feature_type}")

