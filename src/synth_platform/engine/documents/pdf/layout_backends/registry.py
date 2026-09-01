"""Maps an extraction_method string to a LayoutBackend class.

Adding a new layout backend (e.g. a DoclingBackend or an ML layout model)
means registering a new class here - it must never require changes to
run_template_compilation(), which only ever sees the generic LayoutBackend
interface. Mirrors src/synthesis/registry.py and src/adapters/registry.py.
"""

from __future__ import annotations

from synth_platform.engine.documents.pdf.layout_backends.base import LayoutBackend
from synth_platform.engine.documents.pdf.layout_backends.docling_backend import DoclingLayoutBackend
from synth_platform.engine.documents.pdf.layout_backends.native_backend import NativeLayoutBackend
from synth_platform.engine.documents.pdf.layout_backends.ocr_backend import OcrLayoutBackend

_BACKEND_REGISTRY: dict[str, type[LayoutBackend]] = {
    "docling": DoclingLayoutBackend,
    "native": NativeLayoutBackend,
    "ocr": OcrLayoutBackend,
}


class UnknownLayoutBackendError(Exception):
    """Raised when extraction_method does not match a registered LayoutBackend."""


def get_layout_backend_class(backend_name: str) -> type[LayoutBackend]:
    backend_cls = _BACKEND_REGISTRY.get(backend_name)
    if backend_cls is None:
        known = ", ".join(sorted(_BACKEND_REGISTRY)) or "(none registered)"
        raise UnknownLayoutBackendError(f"Unknown layout backend {backend_name!r}. Known backends: {known}.")
    return backend_cls


def resolve_layout_backend(backend_name: str) -> LayoutBackend:
    return get_layout_backend_class(backend_name)()
