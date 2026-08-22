"""Evidence tracking for LLM text generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TextGenerationEvidence:
    table: str
    column: str
    rows_requested: int = 0
    rows_generated: int = 0
    llm_used: bool = False
    fallback_count: int = 0
    privacy_fail_count: int = 0
    format_fail_count: int = 0
    source_replay_count: int = 0
    average_word_count: float = 0.0
    warnings: List[str] = field(default_factory=list)
    generation_time_seconds: float = 0.0
    llm_rows_attempted: int = 0
    cache_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def merge_evidence(items: List[TextGenerationEvidence]) -> Dict[str, Any]:
    if not items:
        return {"columns": [], "summary": {}}
    return {
        "columns": [item.to_dict() for item in items],
        "summary": {
            "column_count": len(items),
            "total_rows_generated": sum(i.rows_generated for i in items),
            "total_fallback_count": sum(i.fallback_count for i in items),
            "total_privacy_fail_count": sum(i.privacy_fail_count for i in items),
            "llm_used_any": any(i.llm_used for i in items),
        },
    }
