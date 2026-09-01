"""Domain vocabulary presets — plugin/config driven, not hardcoded in core path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class DomainVocabulary:
    name: str
    context_hint: str
    templates: Sequence[str] = field(default_factory=list)
    topic_words: Sequence[str] = field(default_factory=list)


_REGISTRY: Dict[str, DomainVocabulary] = {}


def register_domain_vocabulary(vocab: DomainVocabulary) -> None:
    _REGISTRY[vocab.name.lower()] = vocab


def get_domain_vocabulary(name: Optional[str]) -> DomainVocabulary:
    key = str(name or "generic").lower()
    return _REGISTRY.get(key, _REGISTRY["generic"])


def _bootstrap_defaults() -> None:
    if _REGISTRY:
        return
    register_domain_vocabulary(
        DomainVocabulary(
            name="generic",
            context_hint="Generic business and operational workflow language.",
            templates=(
                "Routine review note documents a standard follow-up action for the generated record.",
                "Synthetic narrative summarizes a non-sensitive operational update without private identifiers.",
                "Support-style note captures a neutral status update aligned with the provided context.",
                "General feedback text describes a routine workflow step in plain language.",
            ),
            topic_words=("review", "support", "workflow", "status", "follow", "update", "context"),
        )
    )
    register_domain_vocabulary(
        DomainVocabulary(
            name="ecommerce",
            context_hint="E-commerce order, fulfillment, and customer support language.",
            templates=(
                "Order review note mentions shipment timing and routine fulfillment follow up.",
                "Customer support note documents a standard delivery or return workflow update.",
                "Synthetic comment summarizes a non-sensitive order handling step.",
            ),
            topic_words=("order", "shipment", "return", "fulfillment", "delivery", "support"),
        )
    )
    register_domain_vocabulary(
        DomainVocabulary(
            name="saas",
            context_hint="SaaS subscription, onboarding, and ticket workflow language.",
            templates=(
                "Ticket note records a routine onboarding or configuration follow up.",
                "Synthetic summary describes a standard account workflow update.",
                "Support message documents a non-sensitive product usage review.",
            ),
            topic_words=("subscription", "onboarding", "ticket", "workspace", "feature", "support"),
        )
    )
    register_domain_vocabulary(
        DomainVocabulary(
            name="healthcare",
            context_hint="Healthcare operations language without clinical identifiers or diagnoses.",
            templates=(
                "Administrative note records a routine scheduling or intake follow up.",
                "Synthetic case note describes a non-clinical workflow update.",
            ),
            topic_words=("intake", "scheduling", "referral", "administrative", "follow", "review"),
        )
    )
    register_domain_vocabulary(
        DomainVocabulary(
            name="education",
            context_hint="Education enrollment and student services language.",
            templates=(
                "Enrollment note documents a routine student services follow up.",
                "Synthetic comment summarizes a standard academic workflow update.",
            ),
            topic_words=("enrollment", "course", "student", "advisor", "registration", "review"),
        )
    )
    register_domain_vocabulary(
        DomainVocabulary(
            name="banking",
            context_hint="Optional banking demo preset — not used unless explicitly selected.",
            templates=(
                "Routine service note documents a standard account review follow up without identifiers.",
                "Synthetic case note describes a non-sensitive mobile wallet settlement workflow update.",
                "Support note records a routine merchant transfer review and standard reconciliation follow up.",
                "Service review note documents a UPI or card settlement workflow without private identifiers.",
            ),
            topic_words=("review", "service", "transfer", "settlement", "support", "workflow", "wallet", "mobile", "bank"),
        )
    )


_bootstrap_defaults()

DOMAIN_VOCABULARIES = _REGISTRY
