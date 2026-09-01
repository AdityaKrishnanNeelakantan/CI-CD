"""Maps a `model_type` string to a SynthesizerAdapter class.

Adding a new synthesizer (CTGAN, TVAE, a different vendor entirely) means
registering a new adapter class here - it must never require changes to
the training service, which only ever sees the generic SynthesizerAdapter
interface. Mirrors src/adapters/registry.py.
"""

from __future__ import annotations

from typing import Any

from synth_platform.engine.training.database.adapters.dp_copula_adapter import DPCopulaSynthesizerAdapter
from synth_platform.engine.training.database.adapters.safe_copula_adapter import SafeCopulaSynthesizerAdapter
from synth_platform.engine.training.database.base import SynthesisError, SynthesizerAdapter

_ADAPTER_REGISTRY: dict[str, type[SynthesizerAdapter]] = {
    "safe_gaussian_copula": SafeCopulaSynthesizerAdapter,
    "dp_gaussian_copula": DPCopulaSynthesizerAdapter,
}

_LAZY_ADAPTERS = {"sdv_gaussian_copula"}


def get_synthesizer_adapter_class(model_type: str) -> type[SynthesizerAdapter]:
    if model_type == "sdv_gaussian_copula":
        try:
            from synth_platform.engine.training.database.adapters.sdv_adapter import SDVSynthesizerAdapter
        except Exception as exc:
            raise SynthesisError(
                "SDV Gaussian Copula is unavailable in this environment. "
                "Use safe_gaussian_copula or fix the installed sdv package."
            ) from exc
        return SDVSynthesizerAdapter

    adapter_cls = _ADAPTER_REGISTRY.get(model_type)
    if adapter_cls is None:
        known = ", ".join(sorted([*_ADAPTER_REGISTRY, *_LAZY_ADAPTERS])) or "(none registered)"
        raise SynthesisError(f"Unknown model_type {model_type!r}. Known types: {known}.")
    return adapter_cls


def create_synthesizer_adapter(model_type: str, **adapter_kwargs: Any) -> SynthesizerAdapter:
    """adapter_kwargs are forwarded to the adapter class's constructor -
    e.g. epsilon_budget/column_bounds for "dp_gaussian_copula",
    category_minimum_support for "safe_gaussian_copula". A model_type
    that doesn't accept a given kwarg raises the constructor's own
    TypeError rather than silently ignoring it.
    """
    return get_synthesizer_adapter_class(model_type)(**adapter_kwargs)
