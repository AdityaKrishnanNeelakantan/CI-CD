"""SDV (GaussianCopulaSynthesizer) implementation of SynthesizerAdapter.

All SDV-specific behaviour - sdtype mapping, metadata construction, the
private _set_random_state seeding mechanism - is isolated in this module.
Nothing outside this file should know it is talking to SDV.

Metadata is always built explicitly, column by column, from the approved
dataset_contract - never via SDV's own detect_from_dataframe/
detect_from_csv auto-detection. Letting SDV re-guess column types would
violate "model decisions must not be inferred again inside the training
service": the approved semantic_type is already a settled fact by the
time training happens, and re-guessing from raw dtypes reintroduces
exactly the "numeric ID modelled as a continuous quantity" trap the
semantic inference stage exists to prevent (verified empirically: SDV's
own auto-detection produced nonsensical multi-million-value "ages" on a
column we know is bounded 18-100).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from synth_platform.engine.common.database.privacy.context_fields import CONTEXT_AWARE_KINDS, ContextFieldKind, resolve_context_kind
from synth_platform.engine.training.database.base import SynthesisError, SynthesizerAdapter

APPROVED_STATUS = "approved"

SEMANTIC_TO_SDTYPE = {
    "numerical": "numerical",
    "category": "categorical",
    "boolean": "boolean",
    "datetime": "datetime",
    "email": "email",
    "identifier": "id",
    "person_name": "name",
    # RDT warns that sdtype 'text' is deprecated in favor of 'id', but SDV
    # rejects pii=True on 'id'. Keep 'text'+pii for free_text until SDV offers
    # a supported free-text sdtype with PII semantics (see WARNING_AUDIT.md).
    "free_text": "text",
    "phone_number": "phone_number",
}

_KIND_TO_SDTYPE = {
    ContextFieldKind.PERSON_NAME: "name",
    ContextFieldKind.EMAIL: "email",
    ContextFieldKind.PHONE: "phone_number",
    ContextFieldKind.FREE_TEXT: "text",
    ContextFieldKind.ADDRESS: "text",
    ContextFieldKind.CITY: "text",
    ContextFieldKind.STATE: "text",
    ContextFieldKind.POSTAL_CODE: "text",
    ContextFieldKind.COUNTRY: "text",
    ContextFieldKind.COMPANY: "text",
    ContextFieldKind.IDENTIFIER: "id",
    ContextFieldKind.UUID: "id",
}


class SDVSynthesizerAdapter(SynthesizerAdapter):
    model_type = "sdv_gaussian_copula"
    #: cloudpickle - RESTRICTED: deserializing this file can execute
    #: arbitrary code. Only load a .pkl from a trusted, access-controlled
    #: environment. See src/synthesis/adapters/safe_copula_adapter.py for
    #: a pickle-free alternative for this same statistical model.
    file_extension = ".pkl"
    serialization_format = "cloudpickle"

    def __init__(self) -> None:
        self._synthesizer: GaussianCopulaSynthesizer | None = None
        self._trained_columns: list[str] = []
        self._seed: int | None = None

    def fit(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_contract: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        columns_contract = table_contract.get("columns", {})
        primary_keys = table_contract.get("primary_key", [])

        trainable: dict[str, Any] = {}
        excluded: list[dict[str, str]] = []
        for column_name, column_contract in columns_contract.items():
            if column_contract.get("inference_status") != APPROVED_STATUS:
                excluded.append(
                    {
                        "column": column_name,
                        "reason": f"inference_status={column_contract.get('inference_status')}",
                    }
                )
                continue
            semantic_type = column_contract["semantic_type"]
            if semantic_type not in SEMANTIC_TO_SDTYPE:
                excluded.append({"column": column_name, "reason": f"unmapped_semantic_type={semantic_type}"})
                continue
            trainable[column_name] = column_contract

        if not trainable:
            raise SynthesisError(f"table {table_name!r} has no trainable approved columns")

        missing = set(trainable) - set(df.columns)
        if missing:
            raise SynthesisError(f"training data is missing contract columns: {sorted(missing)}")

        training_df = df[list(trainable.keys())].copy()

        single_pk = primary_keys[0] if len(primary_keys) == 1 and primary_keys[0] in trainable else None
        if len(primary_keys) > 1:
            excluded.append(
                {
                    "column": ",".join(primary_keys),
                    "reason": "composite_primary_key_not_supported_by_this_adapter",
                }
            )

        metadata = Metadata()
        metadata.add_table(table_name)
        for column_name, column_contract in trainable.items():
            semantic_type = column_contract["semantic_type"]
            kind = resolve_context_kind(column_name, semantic_type)
            if column_name == single_pk:
                sdtype = "id"
            elif kind in CONTEXT_AWARE_KINDS and kind in _KIND_TO_SDTYPE:
                # Name-cue / PII kinds must never be modelled as ordinary
                # categoricals even if the contract mis-labelled them.
                sdtype = _KIND_TO_SDTYPE[kind]
            else:
                sdtype = SEMANTIC_TO_SDTYPE[semantic_type]
            kwargs: dict[str, Any] = {}
            if sdtype in ("email", "name", "text", "phone_number") or kind in CONTEXT_AWARE_KINDS:
                # pii=True: SDV generates fake/placeholder values and never
                # models the real emails/names/free-text statistically -
                # real values never influence the synthetic output for
                # this column. Verified empirically (not just per SDV's
                # docs): fit+sample on a column of unique real sentences
                # produced zero overlap with the source values.
                # Note: SDV rejects pii=True on sdtype 'id'; identifiers rely
                # on pre-input masking + id generation instead.
                if sdtype != "id":
                    kwargs["pii"] = True
            metadata.add_column(column_name, sdtype=sdtype, table_name=table_name, **kwargs)

        if single_pk:
            metadata.set_primary_key(single_pk, table_name=table_name)

        # An "identifier" semantic type does not by itself mean the column
        # is unique within this table - a foreign key referencing another
        # table's primary key is semantically an identifier too, but
        # legitimately repeats (e.g. orders.customer_id). SDV's alternate
        # keys must be unique, so only genuinely unique identifier columns
        # are declared as one; a repeating "identifier" column still gets
        # sdtype="id" (still Faker-generated, never modelled from real
        # values) but without the uniqueness constraint. Real cross-table
        # referential integrity is out of scope for this single-table
        # adapter (a later multi-table checkpoint).
        alternate_keys = [
            name
            for name, col in trainable.items()
            if col["semantic_type"] == "identifier"
            and name != single_pk
            and training_df[name].is_unique
        ]
        if alternate_keys:
            metadata.add_alternate_keys(alternate_keys, table_name=table_name)

        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(training_df)

        self._synthesizer = synthesizer
        self._trained_columns = list(trainable.keys())
        self._seed = seed

        return {
            "model_type": self.model_type,
            "trained_columns": self._trained_columns,
            "excluded_columns": excluded,
            "row_count": len(training_df),
            "primary_key": single_pk,
            "alternate_keys": alternate_keys,
        }

    def sample(self, num_rows: int, seed: int | None = None) -> pd.DataFrame:
        # category_overrides is deliberately not in this signature (unlike
        # the ABC's other concrete adapters) - its fitted transformer state
        # has no introspectable per-category boundaries, so a caller passing
        # it gets the base class's own documented TypeError rather than a
        # custom exception type; use safe_gaussian_copula or
        # dp_gaussian_copula for this feature instead.
        if self._synthesizer is None:
            raise SynthesisError("sample() called before fit()/load()")
        if num_rows <= 0:
            raise ValueError(f"num_rows must be positive, got {num_rows}")

        effective_seed = self._seed if seed is None else seed
        if effective_seed is not None:
            # Private API: SDV does not expose a public per-call seed
            # parameter for GaussianCopulaSynthesizer.sample() (verified
            # empirically - the public numpy global seed is ignored once
            # SDV fixes its own internal RNG state on first sample()).
            # Isolated here so only this adapter breaks if a future SDV
            # version removes it, not the rest of the application.
            self._synthesizer._set_random_state(effective_seed)

        return self._synthesizer.sample(num_rows)

    def save(self, path: str | Path) -> None:
        if self._synthesizer is None:
            raise SynthesisError("save() called before fit()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._synthesizer.save(str(path))
        # SDV's own get_metadata() returns a differently-shaped object after
        # load() than after fit() (verified empirically - the post-load
        # object's to_dict() nests columns under "tables"/"table", not at
        # the top level), so trained-column bookkeeping is kept in our own
        # sidecar file rather than depending on that inconsistent internal
        # shape.
        sidecar = self._sidecar_path(path)
        with sidecar.open("w", encoding="utf-8") as f:
            json.dump(
                {"model_type": self.model_type, "trained_columns": self._trained_columns, "seed": self._seed},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> SDVSynthesizerAdapter:
        path = Path(path)
        if not path.is_file():
            raise SynthesisError(f"model file not found: {path}")
        instance = cls()
        try:
            from sdv.utils import load_synthesizer as sdv_load_synthesizer

            instance._synthesizer = sdv_load_synthesizer(str(path))
        except Exception:
            # Fallback for older SDV installs still exposing classmethod load.
            instance._synthesizer = GaussianCopulaSynthesizer.load(str(path))

        sidecar = cls._sidecar_path(path)
        if sidecar.is_file():
            with sidecar.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            instance._trained_columns = meta.get("trained_columns", [])
            instance._seed = meta.get("seed")
        return instance

    @staticmethod
    def _sidecar_path(model_path: Path) -> Path:
        return model_path.with_name(model_path.name + ".meta.json")

