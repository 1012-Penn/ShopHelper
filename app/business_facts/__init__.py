"""Stable business-subject and source adapters.

The corpus remains immutable. Adapters expose namespace-qualified views over
the current factbook and explicitly registered source overlays.
"""

from app.business_facts.order_facts import (
    corpus_order_ids,
    corpus_policy_windows,
    get_corpus_order,
    load_corpus_order_snapshot,
    policy_version_for_date,
)

__all__ = [
    "corpus_order_ids",
    "corpus_policy_windows",
    "get_corpus_order",
    "load_corpus_order_snapshot",
    "policy_version_for_date",
]
