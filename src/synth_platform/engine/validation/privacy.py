"""Privacy metrics grounded in a train/holdout split (DCR ratio, MIA AUC)."""
from __future__ import annotations

import numpy as np


def _min_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.size == 0 or reference.size == 0:
        return np.full(len(query), np.inf)
    q2 = np.einsum("ij,ij->i", query, query)[:, None]
    r2 = np.einsum("ij,ij->i", reference, reference)[None, :]
    out = np.empty(len(query))
    for s in range(0, len(query), 512):
        e = min(s + 512, len(query))
        d2 = q2[s:e] + r2 - 2.0 * (query[s:e] @ reference.T)
        np.maximum(d2, 0.0, out=d2)
        out[s:e] = np.sqrt(d2.min(axis=1))
    return out


def dcr_stats(synthetic, train, holdout):
    """Return (ratio, near_dup_fraction).

    ratio = median(synthetic->train) / median(holdout->train). Informational:
    an empirical-marginal sampler produces central points, so ratio < 1 is
    expected and benign on its own. The memorization signal is near_dup_fraction:
    the share of synthetic rows sitting essentially on top of a training row
    (distance below a tiny fraction of the holdout's own closest-record scale).
    """
    d_syn = _min_distances(synthetic, train)
    d_hold = _min_distances(holdout, train)
    med_syn = float(np.median(d_syn)) if d_syn.size else 0.0
    med_hold = float(np.median(d_hold)) if d_hold.size else 1.0
    ratio = med_syn / med_hold if med_hold > 1e-12 else (1.0 if med_syn > 1e-12 else 0.0)
    # Memorization signal: synthetic rows that land closer to train than the
    # holdout's OWN closest real records do (below the 1st percentile of
    # holdout->train distances). Scale-free; targets copying, not central-tendency.
    if d_hold.size:
        floor = float(np.percentile(d_hold, 1))
        near_dup = float(np.mean(d_syn < floor)) if d_syn.size else 0.0
    else:
        near_dup = 0.0
    return ratio, near_dup


def _auc(scores, labels):
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def membership_inference_auc(synthetic, train, holdout):
    if synthetic.size == 0:
        return 0.5
    d_tr = _min_distances(train, synthetic)
    d_ho = _min_distances(holdout, synthetic)
    scores = np.concatenate([-d_tr, -d_ho])
    labels = np.concatenate([np.ones(len(d_tr)), np.zeros(len(d_ho))])
    return _auc(scores, labels)
