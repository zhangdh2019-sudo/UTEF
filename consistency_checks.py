from __future__ import annotations

from typing import Any, Dict

import torch


def check_temporal_boundaries(split) -> None:
    """
    Ensure no timestamp overlap across train/val/test.
    """
    if split.train_timestamp.numel() > 0 and split.val_timestamp.numel() > 0:
        assert int(split.train_timestamp.max().item()) < int(split.val_timestamp.min().item()), "train/val timestamp overlap"
    if split.val_timestamp.numel() > 0 and split.test_timestamp.numel() > 0:
        assert int(split.val_timestamp.max().item()) < int(split.test_timestamp.min().item()), "val/test timestamp overlap"


def check_label_matches_rating(split) -> None:
    """
    Validate labels are derived from rating sign:
      label=1 iff rating>0, label=0 iff rating<0
    """
    def _check(rating: torch.Tensor, label: torch.Tensor, name: str):
        if rating.numel() == 0:
            return
        # Allow a tiny epsilon
        trust = rating > 0
        distrust = rating < 0
        assert (label[trust] == 1).all(), f"{name}: rating>0 edges must have label=1"
        assert (label[distrust] == 0).all(), f"{name}: rating<0 edges must have label=0"
        # rating==0 should be absent after parsing
        zero_mask = rating == 0
        assert int(zero_mask.sum().item()) == 0, f"{name}: rating==0 edges should be filtered"

    _check(split.train_edge_rating, split.train_edge_label, "train")
    _check(split.val_edge_rating, split.val_edge_label, "val")
    _check(split.test_edge_rating, split.test_edge_label, "test")


def check_feature_type_constraints(feature_type: str, features, expected_dim: int, node2vec_prefix: str = "n2v_") -> None:
    feature_type = feature_type.lower()
    assert features.feature_type == feature_type, "feature_type mismatch"
    assert features.feature_dim == expected_dim, f"{feature_type}: unexpected feature_dim"
    if feature_type == "node2vec":
        assert all(name.startswith(node2vec_prefix) for name in features.feature_names), "node2vec features must be embedding-only"


def check_threshold_freeze(evaluation_output: Dict[str, Any]) -> None:
    thr = evaluation_output.get("best_threshold", None)
    assert thr is not None, "evaluation output missing best_threshold"
    assert 0.0 <= thr <= 1.0, "best_threshold must be within [0,1]"

