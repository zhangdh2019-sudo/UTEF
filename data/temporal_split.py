import dataclasses
import os
from typing import Optional, Tuple

import torch

from .bitcoin_otc import (
    TemporalSignedEdges,
    build_signed_adjacency,
    load_bitcoin_alpha,
    load_bitcoin_otc,
    load_wiki_rfa,
)


@dataclasses.dataclass
class TemporalSignedSplit:
    dataset_name: str
    num_nodes: int
    # Each split edge index: [2, num_edges]
    train_edge_index: torch.Tensor
    val_edge_index: torch.Tensor
    test_edge_index: torch.Tensor
    # Edge labels: 1 trust, 0 distrust
    train_edge_label: torch.Tensor
    val_edge_label: torch.Tensor
    test_edge_label: torch.Tensor
    # Edge ratings from raw dataset (float), used for consistency checks
    train_edge_rating: torch.Tensor
    val_edge_rating: torch.Tensor
    test_edge_rating: torch.Tensor
    # Timestamps (int64)
    train_timestamp: torch.Tensor
    val_timestamp: torch.Tensor
    test_timestamp: torch.Tensor

    # Train-visible signed adjacency (normalized, sparse)
    A_pos: torch.Tensor
    A_neg: torch.Tensor

    # Seen nodes are endpoints of train edges (inductive OOD breakdown)
    seen_node_mask: torch.Tensor  # [num_nodes] bool

    # For strictness/debug
    train_end_ts: int
    val_end_ts: int

    # Train-only temporal statistics for temporal encoding
    # Node-level: last incident timestamp within train edges
    node_last_ts: torch.Tensor  # [num_nodes] long, -1 means no history
    # Pair-level directed history keyed by (u,v) -> last timestamp/counts in train
    pair_keys_sorted: torch.Tensor  # [num_pairs] long, sorted keys
    pair_pos_count_sorted: torch.Tensor  # [num_pairs] long
    pair_neg_count_sorted: torch.Tensor  # [num_pairs] long
    pair_last_ts_sorted: torch.Tensor  # [num_pairs] long, -1 means no history


def _strict_ratio_timestamp_split(
    timestamps: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    Split by timestamp with *no overlap* across splits.

    We choose cut timestamps based on cumulative sorted order indices,
    then expand cut to the boundary of unique timestamps so that:
      max(train_ts) < min(val_ts) <= max(val_ts) < min(test_ts)
    """
    assert 0.0 < train_ratio < 1.0
    assert 0.0 < val_ratio < 1.0

    E = timestamps.numel()
    sorted_ts, sorted_idx = torch.sort(timestamps)
    # initial target indices by ratio
    train_target = int(E * train_ratio)
    val_target = int(E * (train_ratio + val_ratio))

    train_target = max(0, min(train_target, E - 1))
    val_target = max(0, min(val_target, E - 1))

    # Pick cut timestamps: expand to ensure strict boundaries
    train_cut_ts = int(sorted_ts[train_target].item())
    val_cut_ts = int(sorted_ts[val_target].item())

    # Train = ts <= train_cut_ts
    train_mask_sorted = sorted_ts <= train_cut_ts
    # Val should start strictly after train_cut_ts
    # Set val_end_ts as the last timestamp <= val_cut_ts but > train_cut_ts.
    val_mask_sorted = (sorted_ts > train_cut_ts) & (sorted_ts <= val_cut_ts)

    # Remaining are test
    test_mask_sorted = ~(train_mask_sorted | val_mask_sorted)

    train_idx = sorted_idx[train_mask_sorted]
    val_idx = sorted_idx[val_mask_sorted]
    test_idx = sorted_idx[test_mask_sorted]

    train_end_ts = train_cut_ts
    val_end_ts = val_cut_ts
    return train_idx, val_idx, test_idx, train_end_ts, val_end_ts


def temporal_signed_trust_split(
    edges: TemporalSignedEdges,
    dataset_name: str = "bitcoin_otc",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    make_symmetric_adj: bool = False,
) -> TemporalSignedSplit:
    """
    Temporal signed trust prediction split.

    Important design choice (per your requirement):
      - Adjacency for propagation uses ONLY train edges.
      - val/test edges are labels to predict trust/distrust.
    """
    if edges.edge_index.size(1) != edges.edge_label.size(0):
        raise ValueError("Edge index and labels size mismatch")
    if edges.timestamp.size(0) != edges.edge_label.size(0):
        raise ValueError("Timestamp and labels size mismatch")

    num_nodes = int(edges.edge_index.max().item()) + 1
    timestamps = edges.timestamp

    train_idx, val_idx, test_idx, train_end_ts, val_end_ts = _strict_ratio_timestamp_split(
        timestamps=timestamps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # Extract split edges
    train_edge_index = edges.edge_index[:, train_idx]
    val_edge_index = edges.edge_index[:, val_idx]
    test_edge_index = edges.edge_index[:, test_idx]

    train_edge_label = edges.edge_label[train_idx]
    val_edge_label = edges.edge_label[val_idx]
    test_edge_label = edges.edge_label[test_idx]

    train_edge_rating = edges.edge_rating[train_idx]
    val_edge_rating = edges.edge_rating[val_idx]
    test_edge_rating = edges.edge_rating[test_idx]

    train_timestamp = edges.timestamp[train_idx]
    val_timestamp = edges.timestamp[val_idx]
    test_timestamp = edges.timestamp[test_idx]

    # Strictness checks
    if train_timestamp.numel() > 0 and val_timestamp.numel() > 0:
        if int(train_timestamp.max().item()) >= int(val_timestamp.min().item()):
            raise RuntimeError("Non-strict boundary: train and val share timestamps")
    if val_timestamp.numel() > 0 and test_timestamp.numel() > 0:
        if int(val_timestamp.max().item()) >= int(test_timestamp.min().item()):
            raise RuntimeError("Non-strict boundary: val and test share timestamps")

    # Train-only signed adjacency
    A_pos, A_neg = build_signed_adjacency(
        num_nodes=num_nodes,
        edge_index=train_edge_index,
        edge_label=train_edge_label,
        device=torch.device("cpu"),
        make_symmetric=make_symmetric_adj,
    )

    # Seen nodes for OOD breakdown
    seen_nodes = torch.zeros(num_nodes, dtype=torch.bool)
    seen_nodes[train_edge_index.view(-1)] = True

    # ---- Precompute temporal encoding stats (train-only) ----
    src_train = train_edge_index[0]
    dst_train = train_edge_index[1]
    t_train = train_timestamp
    y_train = train_edge_label

    # Node last incident time (max over src/dst incident)
    node_last_ts = torch.full((num_nodes,), -1, dtype=torch.long)
    incident_nodes = torch.cat([src_train, dst_train], dim=0)
    incident_times = torch.cat([t_train, t_train], dim=0)
    # scatter_reduce_ exists on modern torch; fallback if not.
    try:
        node_last_ts.scatter_reduce_(0, incident_nodes, incident_times, reduce="amax", include_self=True)
    except Exception:
        # Fallback: CPU loop (still fine for ~6k nodes)
        for n in torch.unique(incident_nodes):
            mask = incident_nodes == n
            node_last_ts[n] = int(incident_times[mask].max().item())

    # Pair stats (u->v) from train edges
    train_pair_keys = (src_train.to(torch.int64) * num_nodes + dst_train.to(torch.int64)).to(torch.int64)
    unique_keys, inv = torch.unique(train_pair_keys, return_inverse=True)
    num_pairs = unique_keys.numel()

    pos_count = torch.zeros(num_pairs, dtype=torch.long)
    neg_count = torch.zeros(num_pairs, dtype=torch.long)
    pos_mask = y_train == 1
    neg_mask = y_train == 0
    pos_count.scatter_add_(0, inv[pos_mask], torch.ones_like(inv[pos_mask], dtype=torch.long))
    neg_count.scatter_add_(0, inv[neg_mask], torch.ones_like(inv[neg_mask], dtype=torch.long))

    pair_last_ts = torch.full((num_pairs,), -1, dtype=torch.long)
    try:
        pair_last_ts.scatter_reduce_(0, inv, t_train.to(torch.long), reduce="amax", include_self=True)
    except Exception:
        for k in range(num_pairs):
            m = inv == k
            pair_last_ts[k] = int(t_train[m].max().item()) if int(m.sum().item()) > 0 else -1

    sort_perm = torch.argsort(unique_keys)
    pair_keys_sorted = unique_keys[sort_perm]
    pair_pos_count_sorted = pos_count[sort_perm]
    pair_neg_count_sorted = neg_count[sort_perm]
    pair_last_ts_sorted = pair_last_ts[sort_perm]

    return TemporalSignedSplit(
        dataset_name=dataset_name,
        num_nodes=num_nodes,
        train_edge_index=train_edge_index,
        val_edge_index=val_edge_index,
        test_edge_index=test_edge_index,
        train_edge_label=train_edge_label,
        val_edge_label=val_edge_label,
        test_edge_label=test_edge_label,
        train_edge_rating=train_edge_rating,
        val_edge_rating=val_edge_rating,
        test_edge_rating=test_edge_rating,
        train_timestamp=train_timestamp,
        val_timestamp=val_timestamp,
        test_timestamp=test_timestamp,
        A_pos=A_pos,
        A_neg=A_neg,
        seen_node_mask=seen_nodes,
        train_end_ts=train_end_ts,
        val_end_ts=val_end_ts,
        node_last_ts=node_last_ts,
        pair_keys_sorted=pair_keys_sorted,
        pair_pos_count_sorted=pair_pos_count_sorted,
        pair_neg_count_sorted=pair_neg_count_sorted,
        pair_last_ts_sorted=pair_last_ts_sorted,
    )


def build_temporal_bitcoin_otc(
    root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> TemporalSignedSplit:
    edges = load_bitcoin_otc(root=root)
    return temporal_signed_trust_split(
        edges=edges,
        dataset_name="bitcoin_otc",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


def build_temporal_dataset(
    dataset: str,
    data_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    make_symmetric_adj: bool = False,
) -> TemporalSignedSplit:
    dataset_key = dataset.strip().lower()
    root_abs = os.path.abspath(data_root)
    if os.path.basename(root_abs).lower() == dataset_key:
        dataset_root = root_abs
    else:
        dataset_root = os.path.join(root_abs, dataset_key)

    if dataset_key == "bitcoin_otc":
        edges = load_bitcoin_otc(root=dataset_root)
    elif dataset_key == "bitcoin_alpha":
        edges = load_bitcoin_alpha(root=dataset_root)
    elif dataset_key == "wiki_rfa":
        edges = load_wiki_rfa(root=dataset_root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")

    return temporal_signed_trust_split(
        edges=edges,
        dataset_name=dataset_key,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        make_symmetric_adj=make_symmetric_adj,
    )

