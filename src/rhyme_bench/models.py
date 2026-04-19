"""Data models for incidents, queries, and evaluation results."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .taxonomy import CauseClass, ConfusabilityTier


# ---------------------------------------------------------------------------
# Fingerprint — hidden label used for ground-truth construction only
# ---------------------------------------------------------------------------

class Fingerprint(BaseModel):
    """Structured fingerprint for ground-truth labeling. Not visible to models under test."""

    # Golden signals
    latency_shift: str = Field(description="e.g. 'uniform_increase', 'bimodal', 'p99_spike'")
    error_rate_pattern: str = Field(description="e.g. 'step_increase', 'gradual', 'bursty'")
    traffic_correlation: str = Field(description="e.g. 'correlated', 'uncorrelated', 'inverse'")

    # Topology
    origin_service: str = Field(description="Service where the proximal cause lives")
    affected_services: list[str] = Field(description="Services showing symptoms")
    topology_pattern: str = Field(description="e.g. 'single_service', 'fan_out', 'cascade'")

    # Temporal
    onset_shape: str = Field(description="e.g. 'step', 'ramp', 'sawtooth'")
    duration_pattern: str = Field(description="e.g. 'persistent', 'self_resolving', 'periodic'")
    deploy_correlation: bool = Field(description="Did onset coincide with a deployment?")

    # Amplification
    amplification_ratio: float = Field(
        description="Ratio of downstream request rate to inbound rate. >1 suggests retry storm."
    )

    # DB-side metrics (§13 Q3 — synthesized to discriminate DB slow query vs pool exhaustion)
    db_query_time_ms: float | None = Field(
        default=None,
        description="Average DB query execution time in ms. Elevated for slow queries, normal for pool issues.",
    )
    db_connection_pool_utilization: float | None = Field(
        default=None,
        description="Connection pool utilization 0-1. Near 1.0 for pool exhaustion, normal for slow queries.",
    )
    db_lock_contention: float | None = Field(
        default=None,
        description="Lock contention ratio 0-1. Elevated for data contention, race conditions.",
    )

    # Deploy instrumentation (§13 Q4 — deploy.code vs deploy.config distinction)
    deploy_type: str | None = Field(
        default=None,
        description="'code', 'config', or None. Discriminates code regression from config regression.",
    )


# ---------------------------------------------------------------------------
# Incident — the core data record
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    timestamp: str
    severity: str  # critical, warning, info
    service: str
    message: str


class LogLine(BaseModel):
    timestamp: str
    service: str
    level: str  # ERROR, WARN, INFO, DEBUG
    message: str


class TopologyNode(BaseModel):
    service: str
    kind: str = "http"  # http, grpc, database, cache, queue


class TopologyEdge(BaseModel):
    source: str
    target: str
    protocol: str = "http"


class TopologyFragment(BaseModel):
    nodes: list[TopologyNode]
    edges: list[TopologyEdge]


class IncidentPayload(BaseModel):
    """What the model-under-test sees."""

    incident_id: str
    summary: str = Field(description="Responder summary, 2-5 sentences")
    alerts: list[Alert]
    log_lines: list[LogLine]
    topology: TopologyFragment
    incident_start: str
    incident_end: str | None = None

    def to_text(self, include_topology: bool = True) -> str:
        """Flatten to a single text string for retrieval and analysis."""
        parts = [self.summary]
        for alert in self.alerts:
            parts.append(f"[{alert.severity}] {alert.service}: {alert.message}")
        for log in self.log_lines:
            parts.append(f"[{log.level}] {log.service}: {log.message}")
        if include_topology:
            for edge in self.topology.edges:
                parts.append(f"{edge.source} -> {edge.target}")
        return "\n".join(parts)


class IncidentLabels(BaseModel):
    """Hidden ground-truth labels. Not visible to models under test."""

    cause_class: CauseClass
    confusability_tier: ConfusabilityTier
    remediation_canonical: str
    remediation_masks_symptom: str
    remediation_would_worsen: str | None = None
    fingerprint: Fingerprint


class Incident(BaseModel):
    """Complete incident record = payload + labels."""

    payload: IncidentPayload
    labels: IncidentLabels


# ---------------------------------------------------------------------------
# Corpus and query set
# ---------------------------------------------------------------------------

class Corpus(BaseModel):
    version: str
    taxonomy_version: str
    incidents: list[Incident]

    def payloads(self) -> list[IncidentPayload]:
        return [inc.payload for inc in self.incidents]

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> Corpus:
        return cls.model_validate_json(path.read_text())


class QueryGroundTruth(BaseModel):
    """Ground truth for a query incident."""

    query_id: str
    correct_match_ids: list[str]  # same cause class
    tempting_wrong_ids: list[str]  # high symptom similarity, different cause
    confusability_tier: ConfusabilityTier


class QuerySet(BaseModel):
    version: str
    queries: list[QueryGroundTruth]

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> QuerySet:
        return cls.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Model output format (what adapters return)
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    """Token usage for a single model call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RankedMatch(BaseModel):
    incident_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None


class RetrievalResult(BaseModel):
    query_id: str
    ranked_matches: list[RankedMatch]
    token_usage: TokenUsage | None = None


class RemediationChoice(BaseModel):
    """A single option in the multiple-choice remediation task."""

    label: str  # A, B, C, D, E
    text: str
    grade: str = Field(description="'fixed', 'masks_symptom', 'would_worsen', or 'no_op'")


class RemediationQuestion(BaseModel):
    """Multiple-choice remediation question for a query incident."""

    query_id: str
    choices: list[RemediationChoice]
    correct_label: str  # which label is the canonical fix


class RemediationQuestionSet(BaseModel):
    """All remediation questions for the query set."""

    version: str
    questions: list[RemediationQuestion]

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> RemediationQuestionSet:
        return cls.model_validate_json(path.read_text())


class RemediationResult(BaseModel):
    query_id: str
    selected_label: str  # model's chosen label (A-E)
    proposed_remediation: str | None = None  # optional free-text (kept for compatibility)
    token_usage: TokenUsage | None = None
