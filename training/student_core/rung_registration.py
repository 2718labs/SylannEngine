"""rung_registration.py — pre-registration of a gate rung (F1, minimal).

Anti-p-hacking spine: freeze the estimator, baselines, metric, floors, and the autocorrelated-slice
rule BEFORE any performance is computed, and bind the analysis seed to a digest of that frozen
record. A post-hoc change to any registered field changes the digest (and therefore the seed), so a
silently-moved goalpost is detectable. Local hashlib only (never the identity HMAC).

Design of record: docs/superpowers/plans/2026-07-17-v3core-instrument-landing-plan.md (Track C, F1).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RungRegistrationV1:
    """One frozen registration record for one rung of the offline gate ladder."""

    rung_id: str
    primary_estimator: str  # exactly one estimator; a baseline is a thing to beat, not to shop
    feature_blocks: tuple[str, ...]
    baselines: tuple[str, ...]  # e.g. ("persistence", "steelman_ridge", "steelman_gbm")
    metric: str = "paired_valence_arousal_MAE"
    rel_floor: float = 0.15
    abs_floor: float = 0.02
    slice_rule: str = "autocorrelated: |lag1_acf(valence)| >= 0.30"
    ridge_lambda: float = 1.0
    gbm_max_iter: int = 200
    gbm_learning_rate: float = 0.05
    schema_version: int = 1
    notes: tuple[str, ...] = field(default_factory=tuple)

    def canonical_json(self) -> str:
        """Deterministic JSON (sorted keys, fixed separators) for hashing."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def digest(self) -> str:
        """SHA-256 hex of the canonical record. Binds the analysis seed."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def episode_seed(self, base: int = 0) -> int:
        """A split/analysis seed derived from the registration digest (post-hoc change -> new seed)."""
        return (base ^ int(self.digest()[:16], 16)) & 0xFFFFFFFF
