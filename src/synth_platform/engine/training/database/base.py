"""Vendor-agnostic single-table synthesizer contract.

All vendor-specific behaviour (SDV, or any future library) must live only
inside a concrete adapter implementation - the training service and
everything above it must never branch on which library is in use. This is
the same isolation SourceAdapter provides for structured sources
(src/adapters/base.py) and DocumentAdapter provides for documents
(src/documents/base.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd


class SynthesisError(Exception):
    """Raised when training or sampling cannot proceed with the given contract/data."""


class SynthesizerAdapter(ABC):
    #: Short identifier for this adapter, e.g. "sdv_gaussian_copula".
    model_type: str
    #: File extension save()/load() use, e.g. ".pkl" or ".json" - callers
    #: must never hardcode a model file's extension, since it depends on
    #: which adapter (and therefore which serialization format) trained it.
    file_extension: str
    #: "cloudpickle" or "json" - declared explicitly so a privacy reviewer
    #: (src/artifact/manifest.py) never has to infer it from file_extension.
    serialization_format: str

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        table_name: str,
        table_contract: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """Fit the model on already-cleaned, approved-columns-only data.

        Returns fit evidence (trained/excluded columns, primary key, etc.)
        - never raw data.
        """

    @abstractmethod
    def sample(
        self,
        num_rows: int,
        seed: int | None = None,
        category_overrides: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic rows. Must be called after fit() or load().

        category_overrides (column_name -> {category: weight}), where
        supported (src/synthesis/adapters/category_rebalancing.py),
        regenerates a categorical column under a caller-supplied target
        distribution instead of the model's originally learned one,
        without re-fitting anything - "tweak the ratio, no retraining".
        An adapter that cannot support this (e.g. src/synthesis/adapters/
        sdv_adapter.py, which wraps an opaque third-party model) simply
        does not accept this parameter, and a caller passing it gets a
        plain TypeError rather than a silently ignored option.
        """

    def iter_sample(
        self,
        num_rows: int,
        *,
        batch_size: int,
        seed: int | None = None,
        category_overrides: dict[str, dict[str, float]] | None = None,
        row_offset: int = 0,
    ) -> Iterator[pd.DataFrame]:
        """Yield synthetic batches totaling ``num_rows`` (O(batch_size) RAM).

        Default implementation loops :meth:`sample`. Adapters with sequential
        identifier PKs should override to advance ``row_offset`` so IDs stay
        unique across batches.

        Seed continuity: batch ``i`` uses ``seed + i`` when ``seed`` is not None
        (deterministic across runs).
        """
        if num_rows <= 0:
            raise ValueError(f"num_rows must be positive, got {num_rows}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        remaining = num_rows
        batch_index = 0
        offset = int(row_offset)
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            batch_seed = None if seed is None else int(seed) + batch_index
            kwargs: dict[str, Any] = {"seed": batch_seed}
            if category_overrides is not None:
                kwargs["category_overrides"] = category_overrides
            # Prefer adapters that accept row_offset; fall back for SDV etc.
            try:
                frame = self.sample(this_batch, row_offset=offset, **kwargs)  # type: ignore[call-arg]
            except TypeError:
                frame = self.sample(this_batch, **kwargs)
            yield frame.reset_index(drop=True)
            remaining -= this_batch
            offset += this_batch
            batch_index += 1

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model to a single file."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> SynthesizerAdapter:
        """Load a previously-saved fitted model from a single file into a fresh instance."""
