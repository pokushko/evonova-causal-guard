"""Tabu memory archive preventing agents from regenerating falsified causal hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import time
from typing import Callable


def _normalize_tokens(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens for similarity comparisons."""
    return set(re.findall(r"[a-z0-9а-яё]{2,}", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity in [0.0, 1.0]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class TabuEntry:
    """Recorded falsified causal claim with fingerprint and timestamp."""

    fingerprint: str
    statement: str
    treatment: str
    outcome: str
    reason: str
    created_at: float
    tokens: set[str]


class TabuMemory:
    """In-memory Tabu archive blocking exact and near-duplicate falsified claims."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        ttl_seconds: float | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._time = time_fn or time.monotonic
        self._entries: dict[str, TabuEntry] = {}

    def _fingerprint(self, statement: str, treatment: str, outcome: str) -> str:
        norm = f"{treatment.strip().lower()}::{outcome.strip().lower()}::{' '.join(statement.lower().split())}"
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def _prune_expired(self) -> None:
        if self.ttl_seconds is None:
            return
        now = self._time()
        expired = [k for k, v in self._entries.items() if (now - v.created_at) >= self.ttl_seconds]
        for key in expired:
            del self._entries[key]

    def record_falsified(
        self,
        statement: str,
        treatment: str,
        outcome: str,
        reason: str,
    ) -> TabuEntry:
        """Persist a falsified hypothesis into the Tabu archive."""
        self._prune_expired()
        fp = self._fingerprint(statement, treatment, outcome)
        entry = TabuEntry(
            fingerprint=fp,
            statement=statement,
            treatment=treatment,
            outcome=outcome,
            reason=reason,
            created_at=self._time(),
            tokens=_normalize_tokens(statement),
        )
        self._entries[fp] = entry
        return entry

    def check(self, statement: str, treatment: str, outcome: str) -> dict[str, str | float] | None:
        """Return matching Tabu entry details if `statement` matches a falsified claim."""
        self._prune_expired()
        fp = self._fingerprint(statement, treatment, outcome)
        if fp in self._entries:
            e = self._entries[fp]
            return {"match_type": "exact", "fingerprint": e.fingerprint, "reason": e.reason, "similarity": 1.0}

        q_tokens = _normalize_tokens(statement)
        for entry in self._entries.values():
            if entry.treatment.lower() == treatment.lower() and entry.outcome.lower() == outcome.lower():
                sim = _jaccard(q_tokens, entry.tokens)
                if sim >= self.similarity_threshold:
                    return {
                        "match_type": "fuzzy",
                        "fingerprint": entry.fingerprint,
                        "reason": entry.reason,
                        "similarity": round(sim, 4),
                    }
        return None

    def __len__(self) -> int:
        self._prune_expired()
        return len(self._entries)
