import dataclasses
import gzip
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


_BITCOIN_OTC_URL = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
_BITCOIN_ALPHA_URL = "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"
_WIKI_RFA_URL = "https://snap.stanford.edu/data/wiki-RfA.txt.gz"


@dataclasses.dataclass(frozen=True)
class TemporalSignedEdges:
    # edge_index is shape [2, num_edges], 0-indexed
    edge_index: torch.Tensor
    # rating values as float tensor [num_edges]
    edge_rating: torch.Tensor
    # signed trust labels: 1 if rating>0 else 0 if rating<0
    edge_label: torch.Tensor
    # timestamps as integer tensor [num_edges] (seconds since epoch or dataset time)
    timestamp: torch.Tensor


BitcoinOTCEdges = TemporalSignedEdges  # backward-compatible alias


def _download(url: str, dst_gz: str) -> None:
    os.makedirs(os.path.dirname(dst_gz), exist_ok=True)
    if os.path.exists(dst_gz):
        return
    tmp = dst_gz + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst_gz)


def _gunzip(src_gz: str, dst_csv: str) -> None:
    if os.path.exists(dst_csv):
        return
    with gzip.open(src_gz, "rb") as f_in, open(dst_csv, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _parse_signed_csv_rows(dst_csv: str) -> List[Tuple[int, int, float, int]]:
    rows: List[Tuple[int, int, float, int]] = []
    with open(dst_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            u = int(parts[0])
            v = int(parts[1])
            rating = float(parts[2])
            ts = int(float(parts[3]))
            rows.append((u, v, rating, ts))
    return rows


def _build_temporal_edges_from_rows(
    rows: List[Tuple[int, int, float, int]],
    shift_to_zero_index: bool = True,
) -> TemporalSignedEdges:
    if len(rows) == 0:
        raise RuntimeError("No valid temporal signed edges parsed.")

    arr = np.asarray(rows, dtype=np.float64)
    u = arr[:, 0].astype(np.int64)
    v = arr[:, 1].astype(np.int64)
    rating = torch.tensor(arr[:, 2], dtype=torch.float32)
    timestamp = torch.tensor(arr[:, 3].astype(np.int64), dtype=torch.long)

    if shift_to_zero_index:
        min_id = int(min(u.min(), v.min()))
        u = u - min_id
        v = v - min_id

    edge_index = torch.tensor(np.stack([u, v], axis=0), dtype=torch.long)

    trust_mask = rating > 0
    distrust_mask = rating < 0
    keep_mask = trust_mask | distrust_mask
    if keep_mask.sum().item() < rating.numel():
        edge_index = edge_index[:, keep_mask]
        rating = rating[keep_mask]
        timestamp = timestamp[keep_mask]

    edge_label = torch.where(
        rating > 0,
        torch.ones_like(rating, dtype=torch.long),
        torch.zeros_like(rating, dtype=torch.long),
    )
    return TemporalSignedEdges(
        edge_index=edge_index,
        edge_rating=rating,
        edge_label=edge_label,
        timestamp=timestamp,
    )


def _load_snap_signed_csv_dataset(
    root: str,
    url: str,
    dataset_key: str,
    shift_to_zero_index: bool = True,
) -> TemporalSignedEdges:
    """
    Load the raw Bitcoin-OTC signed temporal network.

    CSV format (per SNAP / PyG source):
      u,v,rating,timestamp
    where rating is a signed trust weight in [-10, +10] and timestamp is unix-time.
    """
    root = os.path.abspath(root)
    raw_dir = os.path.join(root, "raw")
    processed_dir = os.path.join(root, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    dst_gz = os.path.join(raw_dir, "soc-sign-bitcoinotc.csv.gz")
    dst_csv = os.path.join(raw_dir, "soc-sign-bitcoinotc.csv")

    _download(url, dst_gz)
    _gunzip(dst_gz, dst_csv)

    # Cache parsed tensors to avoid reparsing.
    cache_path = os.path.join(processed_dir, f"{dataset_key}_edges.pt")
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu")
        return TemporalSignedEdges(
            edge_index=cached["edge_index"],
            edge_rating=cached["edge_rating"],
            edge_label=cached["edge_label"],
            timestamp=cached["timestamp"],
        )

    rows = _parse_signed_csv_rows(dst_csv)
    edges = _build_temporal_edges_from_rows(rows=rows, shift_to_zero_index=shift_to_zero_index)
    torch.save(
        {
            "edge_index": edges.edge_index,
            "edge_rating": edges.edge_rating,
            "edge_label": edges.edge_label,
            "timestamp": edges.timestamp,
        },
        cache_path,
    )
    return edges


def _parse_wiki_rfa_time(dat_str: str) -> int:
    dat_str = dat_str.strip()
    fmts = ("%H:%M, %d %B %Y", "%H:%M, %d %b %Y")
    for fmt in fmts:
        try:
            dt = datetime.strptime(dat_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ValueError(f"Unrecognized DAT format: {dat_str}")


def load_wiki_rfa(root: str, url: str = _WIKI_RFA_URL, shift_to_zero_index: bool = True) -> TemporalSignedEdges:
    """
    Load SNAP Wiki-RfA into temporal signed directed edges.

    Input is block-form text with keys SRC/TGT/VOT/DAT; we map:
      rating = +1 if VOT>0, -1 if VOT<0, ignore VOT==0.
    """
    root = os.path.abspath(root)
    raw_dir = os.path.join(root, "raw")
    processed_dir = os.path.join(root, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    dst_gz = os.path.join(raw_dir, "wiki-RfA.txt.gz")
    dst_txt = os.path.join(raw_dir, "wiki-RfA.txt")
    _download(url, dst_gz)
    _gunzip(dst_gz, dst_txt)

    cache_path = os.path.join(processed_dir, "wiki_rfa_edges.pt")
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu")
        return TemporalSignedEdges(
            edge_index=cached["edge_index"],
            edge_rating=cached["edge_rating"],
            edge_label=cached["edge_label"],
            timestamp=cached["timestamp"],
        )

    user_to_id: Dict[str, int] = {}

    def uid(name: str) -> int:
        name = name.strip()
        if name not in user_to_id:
            user_to_id[name] = len(user_to_id)
        return user_to_id[name]

    rows: List[Tuple[int, int, float, int]] = []
    cur_src = None
    cur_tgt = None
    cur_vot = None
    cur_dat = None

    def flush_record() -> None:
        nonlocal cur_src, cur_tgt, cur_vot, cur_dat
        if cur_src is None or cur_tgt is None or cur_vot is None or cur_dat is None:
            return
        try:
            v = float(cur_vot)
        except Exception:
            return
        if v == 0.0:
            return
        rating = 1.0 if v > 0 else -1.0
        try:
            ts = _parse_wiki_rfa_time(cur_dat)
        except ValueError:
            return
        rows.append((uid(cur_src), uid(cur_tgt), rating, ts))

    with open(dst_txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                flush_record()
                cur_src = cur_tgt = cur_vot = cur_dat = None
                continue
            if line.startswith("SRC:"):
                cur_src = line[4:]
            elif line.startswith("TGT:"):
                cur_tgt = line[4:]
            elif line.startswith("VOT:"):
                cur_vot = line[4:]
            elif line.startswith("DAT:"):
                cur_dat = line[4:]
        flush_record()

    edges = _build_temporal_edges_from_rows(rows=rows, shift_to_zero_index=shift_to_zero_index)
    torch.save(
        {
            "edge_index": edges.edge_index,
            "edge_rating": edges.edge_rating,
            "edge_label": edges.edge_label,
            "timestamp": edges.timestamp,
        },
        cache_path,
    )
    return edges


def load_bitcoin_otc(root: str, url: str = _BITCOIN_OTC_URL, shift_to_zero_index: bool = True) -> TemporalSignedEdges:
    return _load_snap_signed_csv_dataset(
        root=root,
        url=url,
        dataset_key="bitcoin_otc",
        shift_to_zero_index=shift_to_zero_index,
    )


def load_bitcoin_alpha(root: str, url: str = _BITCOIN_ALPHA_URL, shift_to_zero_index: bool = True) -> TemporalSignedEdges:
    return _load_snap_signed_csv_dataset(
        root=root,
        url=url,
        dataset_key="bitcoin_alpha",
        shift_to_zero_index=shift_to_zero_index,
    )


def build_signed_adjacency(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_label: torch.Tensor,
    device: Optional[torch.device] = None,
    make_symmetric: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build normalized signed adjacency for positive and negative edges separately.

    Returns:
      A_pos: sparse tensor [num_nodes, num_nodes]
      A_neg: sparse tensor [num_nodes, num_nodes]
    """
    if device is None:
        device = torch.device("cpu")

    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    labels = edge_label.cpu().numpy()

    pos_mask = labels == 1
    neg_mask = labels == 0

    def _norm_adj(mask: np.ndarray) -> torch.Tensor:
        s = src[mask]
        d = dst[mask]
        if make_symmetric:
            s = np.concatenate([s, d], axis=0)
            d = np.concatenate([d, s[: d.shape[0]]], axis=0)

        # Add self-loop for numerical stability.
        sl = np.arange(num_nodes, dtype=np.int64)
        s = np.concatenate([s, sl], axis=0)
        d = np.concatenate([d, sl], axis=0)

        indices = torch.tensor(np.stack([s, d], axis=0), dtype=torch.long, device=device)
        values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        # For sparse tensor, compute degrees from indices.
        deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        deg.scatter_add_(0, indices[0], values)  # out-degree style
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)

        row_scale = deg_inv_sqrt[indices[0]]
        col_scale = deg_inv_sqrt[indices[1]]
        norm_values = values * row_scale * col_scale
        return torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes), device=device).coalesce()

    A_pos = _norm_adj(pos_mask)
    A_neg = _norm_adj(neg_mask)
    return A_pos, A_neg

