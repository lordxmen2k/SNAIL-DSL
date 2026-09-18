"""Run manifests — structured per-run log of what fired, when, and what it returned.

Every Program.run() emits a manifest. Manifests are JSON-serializable.
They are the audit trail for any SNAIL program — show them to compliance,
to your CEO, to the user who asked why their refund took 2 seconds.
"""

from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ManifestNodeEvent:
    node_name: str
    variant: str  # "ok" or "ood"
    latency_ms: float
    confidence: float | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ManifestErrorEvent:
    node_name: str
    error: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Manifest:
    program_name: str
    run_id: str
    started_at: float
    ended_at: float | None
    total_duration_ms: float
    node_events: list[ManifestNodeEvent]
    errors: list[ManifestErrorEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_name": self.program_name,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_duration_ms": self.total_duration_ms,
            "node_events": [asdict(e) for e in self.node_events],
            "errors": [asdict(e) for e in self.errors],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ManifestBuilder:
    """Accumulates events during a single Program.run() and emits a Manifest."""

    def __init__(self, program_name: str, run_id: str | None = None):
        self.program_name = program_name
        self.run_id = run_id or str(uuid.uuid4())
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._node_events: list[ManifestNodeEvent] = []
        self._errors: list[ManifestErrorEvent] = []

    def record_start(self) -> None:
        self._started_at = time.time()

    def record_end(self) -> None:
        self._ended_at = time.time()

    def record_node(
        self,
        *,
        node_name: str,
        variant: str,
        latency_ms: float,
        confidence: float | None = None,
    ) -> None:
        self._node_events.append(
            ManifestNodeEvent(
                node_name=node_name,
                variant=variant,
                latency_ms=latency_ms,
                confidence=confidence,
            )
        )

    def record_error(self, *, node_name: str, error: str) -> None:
        self._errors.append(
            ManifestErrorEvent(node_name=node_name, error=error)
        )

    def build(self) -> Manifest:
        if self._started_at is None:
            raise RuntimeError("ManifestBuilder.build() called before record_start()")
        ended = self._ended_at if self._ended_at is not None else time.time()
        return Manifest(
            program_name=self.program_name,
            run_id=self.run_id,
            started_at=self._started_at,
            ended_at=self._ended_at,
            total_duration_ms=(ended - self._started_at) * 1000.0,
            node_events=list(self._node_events),
            errors=list(self._errors),
        )
