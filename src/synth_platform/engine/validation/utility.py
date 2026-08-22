"""TSTR utility with numpy models (no sklearn): softmax clf / ridge reg."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from synth_platform.engine.validation.encoding import FeatureEncoder


@dataclass
class UtilityScore:
    ran: bool
    ratio: float
    tstr: float
    trtr: float
    metric: str
    note: str = ""


def _bias(x): return np.hstack([x, np.ones((len(x), 1))])


def _softmax_fit(x, y_idx, k, seed):
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w = rng.normal(0, 0.01, size=(d, k))
    oh = np.zeros((n, k)); oh[np.arange(n), y_idx] = 1.0
    for _ in range(300):
        lo = x @ w; lo -= lo.max(axis=1, keepdims=True)
        e = np.exp(lo); p = e / e.sum(axis=1, keepdims=True)
        w -= 0.1 * (x.T @ (p - oh) / n + 1e-3 * w)
    return w


def _acc(xtr, ytr, xte, yte, classes, seed):
    idx = {c: i for i, c in enumerate(classes)}
    w = _softmax_fit(xtr, np.array([idx[c] for c in ytr]), len(classes), seed)
    inv = {i: c for c, i in idx.items()}
    pred = np.array([inv[p] for p in (xte @ w).argmax(axis=1)], dtype=object)
    return float(np.mean(pred == np.asarray(yte, dtype=object)))


def _r2(xtr, ytr, xte, yte, lam=1.0):
    d = xtr.shape[1]
    beta = np.linalg.solve(xtr.T @ xtr + lam * np.eye(d), xtr.T @ ytr)
    pred = xte @ beta
    ss_res = float(np.sum((yte - pred) ** 2)); ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def tstr_utility(target, kind, train_real, holdout_real, synthetic, exclude, seed=0):
    enc = FeatureEncoder.fit(train_real, exclude=set(exclude) | {target})
    if enc.width == 0:
        return UtilityScore(False, 0, 0, 0, kind, "no usable features")
    xr, xh, xs = _bias(enc.transform(train_real)), _bias(enc.transform(holdout_real)), _bias(enc.transform(synthetic))
    yr, yh, ys = train_real[target], holdout_real[target], synthetic.get(target)
    if ys is None:
        return UtilityScore(False, 0, 0, 0, kind, "target absent in synthetic")
    if kind == "classification":
        classes = sorted(set(yr.dropna().astype(object)) | set(ys.dropna().astype(object)))
        if len(classes) < 2:
            return UtilityScore(False, 0, 0, 0, "accuracy", "single-class target")
        trtr = _acc(xr, yr.astype(object), xh, yh.astype(object), classes, seed)
        tstr = _acc(xs, ys.astype(object), xh, yh.astype(object), classes, seed)
        return UtilityScore(True, tstr / trtr if trtr > 1e-9 else 0.0, tstr, trtr, "accuracy")
    yr_f = pd.to_numeric(yr, errors="coerce").fillna(0).to_numpy(float)
    yh_f = pd.to_numeric(yh, errors="coerce").fillna(0).to_numpy(float)
    ys_f = pd.to_numeric(ys, errors="coerce").fillna(0).to_numpy(float)
    trtr = _r2(xr, yr_f, xh, yh_f)
    if trtr <= 0:
        return UtilityScore(False, 0, trtr, trtr, "r2", "degenerate real baseline")
    tstr = _r2(xs, ys_f, xh, yh_f)
    return UtilityScore(True, max(tstr, 0.0) / trtr, tstr, trtr, "r2")
