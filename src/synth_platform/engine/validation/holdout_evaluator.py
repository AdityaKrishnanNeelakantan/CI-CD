"""Secure holdout evaluator: emits real privacy + utility CheckResults."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from synth_platform.domain.schema.source import SourceConnector
from synth_platform.domain.schema.models import DatabaseSchema
from synth_platform.domain.validation.models import CheckResult, Status
from synth_platform.domain.validation.thresholds import Thresholds
from synth_platform.engine.validation.encoding import FeatureEncoder
from synth_platform.engine.validation.privacy import dcr_stats, membership_inference_auc
from synth_platform.engine.validation.utility import tstr_utility

_T = Thresholds()
_CAP = 1500


@dataclass
class UtilityTask:
    table: str
    target: str
    kind: str = "classification"


@dataclass
class HoldoutSplit:
    train: dict
    holdout: dict
    schema: DatabaseSchema

    def identifier_columns(self, table: str) -> set:
        cols = set()
        ts = self.schema.tables.get(table)
        if ts and ts.primary_key:
            cols.add(ts.primary_key)
        for fk in self.schema.foreign_keys:
            if fk.child_table == table:
                cols.add(fk.child_column)
        return cols


def holdout_split(connector: SourceConnector, holdout_fraction: float = 0.3,
                  seed: int = 0, max_rows: int = 5000) -> HoldoutSplit:
    schema = connector.discover_schema()
    rng = np.random.default_rng(seed)
    train, holdout = {}, {}
    for tname in schema.tables:
        df = connector.sample_table(tname, max_rows, seed).reset_index(drop=True)
        n = len(df)
        if n < 4:
            train[tname], holdout[tname] = df, df.iloc[0:0].copy()
            continue
        perm = rng.permutation(n)
        cut = min(max(int(round(n * (1 - holdout_fraction))), 1), n - 1)
        train[tname] = df.iloc[perm[:cut]].reset_index(drop=True)
        holdout[tname] = df.iloc[perm[cut:]].reset_index(drop=True)
    return HoldoutSplit(train=train, holdout=holdout, schema=schema)


def _cap(x, rng):
    return x if len(x) <= _CAP else x[rng.choice(len(x), _CAP, replace=False)]


class SecureHoldoutEvaluator:
    def __init__(self, split: HoldoutSplit, tasks=None, seed=0):
        self.split, self.tasks, self.seed = split, tasks or [], seed

    def privacy_utility_checks(self, synthetic: dict) -> list:
        return self._privacy(synthetic) + self._utility(synthetic) + [
            CheckResult(name="formal_differential_privacy", dimension="privacy",
                        required=False, ran=False, status=Status.NOT_RUN,
                        detail="no DP mechanism engaged; not claimed")]

    def _privacy(self, synthetic):
        out, rng = [], np.random.default_rng(self.seed)
        for tname, tr in self.split.train.items():
            hold, syn = self.split.holdout.get(tname), synthetic.get(tname)
            if hold is None or syn is None or len(hold) == 0 or len(tr) < 4 or len(syn) == 0:
                out.append(CheckResult(name=f"privacy_dcr[{tname}]", dimension="privacy",
                                       required=True, ran=False, status=Status.NOT_RUN,
                                       detail="insufficient rows for holdout privacy split"))
                continue
            exclude = self.split.identifier_columns(tname)
            enc = FeatureEncoder.fit(tr, exclude=exclude)
            trx, hox, syx = _cap(enc.transform(tr), rng), _cap(enc.transform(hold), rng), _cap(enc.transform(syn), rng)
            ratio, near_dup = dcr_stats(syx, trx, hox)
            st = (Status.PASS if near_dup <= _T.near_dup_max
                  else Status.WARN if near_dup <= 0.10 else Status.FAIL)
            out.append(CheckResult(name=f"privacy_dcr[{tname}]", dimension="privacy",
                                   required=True, ran=True, status=st, metric=float(near_dup),
                                   threshold=_T.near_dup_max,
                                   detail=f"near-duplicate fraction {near_dup:.4f}; "
                                          f"DCR ratio {ratio:.3f} vs holdout (informational)"))
            auc = membership_inference_auc(syx, trx, hox)
            sa = (Status.PASS if auc <= _T.mia_auc_max else Status.WARN if auc <= 0.70 else Status.FAIL)
            out.append(CheckResult(name=f"privacy_mia_auc[{tname}]", dimension="privacy",
                                   required=True, ran=True, status=sa, metric=float(auc),
                                   threshold=_T.mia_auc_max,
                                   detail=f"membership-inference AUC {auc:.3f} (0.5 ideal)"))
        return out

    def _utility(self, synthetic):
        if not self.tasks:
            return [CheckResult(name="tstr_utility", dimension="utility", required=True,
                                ran=False, status=Status.NOT_RUN, detail="no downstream task supplied")]
        out = []
        for task in self.tasks:
            tr, hold, syn = self.split.train.get(task.table), self.split.holdout.get(task.table), synthetic.get(task.table)
            if tr is None or hold is None or syn is None or len(hold) == 0:
                out.append(CheckResult(name=f"tstr[{task.table}.{task.target}]", dimension="utility",
                                       required=True, ran=False, status=Status.NOT_RUN,
                                       detail="missing table or empty holdout"))
                continue
            sc = tstr_utility(task.target, task.kind, tr, hold, syn,
                              exclude=self.split.identifier_columns(task.table), seed=self.seed)
            if not sc.ran:
                out.append(CheckResult(name=f"tstr[{task.table}.{task.target}]", dimension="utility",
                                       required=True, ran=False, status=Status.SKIPPED, detail=sc.note))
                continue
            st = (Status.PASS if sc.ratio >= _T.tstr_ratio_min
                  else Status.WARN if sc.ratio >= 0.60 else Status.FAIL)
            out.append(CheckResult(name=f"tstr[{task.table}.{task.target}]", dimension="utility",
                                   required=True, ran=True, status=st, metric=float(sc.ratio),
                                   threshold=_T.tstr_ratio_min,
                                   detail=f"{sc.metric} TSTR/TRTR = {sc.tstr:.3f}/{sc.trtr:.3f}"))
        return out
