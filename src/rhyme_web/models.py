"""Data models for org-specific incident correlation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Incident import
# ---------------------------------------------------------------------------

class OrgIncident(BaseModel):
    """A real incident from an organization's corpus. Minimal schema."""

    id: str
    summary: str = Field(description="Human-written incident summary, any length")
    timestamp: str = Field(description="ISO 8601 timestamp of incident start")
    severity: str | None = Field(default=None, description="e.g. SEV1, P1, critical")
    service: str | None = Field(default=None, description="Primary affected service")
    url: str | None = Field(default=None, description="Link to incident in company tracker")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra fields")


class OrgCorpus(BaseModel):
    """Imported org incident corpus."""

    incidents: list[OrgIncident]

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> OrgCorpus:
        return cls.model_validate_json(path.read_text())


class ImportResult(BaseModel):
    """Result of importing incidents, including any per-incident errors."""

    corpus: OrgCorpus
    errors: list[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


def parse_incidents(json_text: str) -> ImportResult:
    """Parse incidents from a JSON string, importing what it can.

    Accepts either:
    - A JSON object with an "incidents" array
    - A bare JSON array of incident objects

    Each incident must have at minimum: id, summary, timestamp.
    Invalid incidents are skipped with errors recorded.
    """
    import json

    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if isinstance(raw, list):
        raw_items = raw
    elif isinstance(raw, dict) and "incidents" in raw:
        raw_items = raw["incidents"]
    else:
        raise ValueError(
            "Expected JSON with 'incidents' array or a bare array of incident objects"
        )

    if not isinstance(raw_items, list):
        raise ValueError("'incidents' must be an array")

    incidents: list[OrgIncident] = []
    errors: list[str] = []

    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            errors.append(f"Item {i}: not a JSON object, skipped")
            continue

        item_id = item.get("id", f"(index {i})")

        # Check required fields
        missing = [f for f in ("id", "summary", "timestamp") if f not in item or not item[f]]
        if missing:
            errors.append(f"Item {item_id}: missing required field(s): {', '.join(missing)}")
            continue

        # Coerce types — ensure id, summary, timestamp are strings
        try:
            item["id"] = str(item["id"])
            item["summary"] = str(item["summary"])
            item["timestamp"] = str(item["timestamp"])
        except Exception as e:
            errors.append(f"Item {item_id}: field type error: {e}")
            continue

        # Strip to max reasonable lengths for non-text fields
        if item.get("severity"):
            item["severity"] = str(item["severity"])[:32]
        if item.get("service"):
            item["service"] = str(item["service"])[:255]
        if item.get("url"):
            item["url"] = str(item["url"])

        try:
            inc = OrgIncident.model_validate(item)
            incidents.append(inc)
        except Exception as e:
            errors.append(f"Item {item_id}: validation error: {e}")

    return ImportResult(corpus=OrgCorpus(incidents=incidents), errors=errors)


def import_incidents(path: Path) -> ImportResult:
    """Import incidents from a JSON file. See parse_incidents() for format."""
    return parse_incidents(path.read_text())


# ---------------------------------------------------------------------------
# Pair labeling
# ---------------------------------------------------------------------------

class IncidentPair(BaseModel):
    """A pair of incidents to be labeled by an SRE."""

    pair_id: str
    incident_a_id: str
    incident_b_id: str
    model_confidence: float | None = Field(
        default=None,
        description="Model's correlation confidence for this pair (if available)",
    )
    sampling_bucket: str = Field(
        description="'high_confidence', 'low_confidence', 'medium', or 'random'"
    )


class PairLabel(BaseModel):
    """Human label for an incident pair."""

    pair_id: str
    labeler_id: str
    judgment: str = Field(description="'yes', 'no', or 'maybe'")
    notes: str = ""
    labeled_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class LabelingSession(BaseModel):
    """All pairs and labels for a labeling session."""

    session_id: str
    corpus_path: str
    pairs: list[IncidentPair]
    labels: list[PairLabel] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> LabelingSession:
        return cls.model_validate_json(path.read_text())

    @property
    def progress(self) -> tuple[int, int]:
        labeled_pair_ids = {l.pair_id for l in self.labels}
        return len(labeled_pair_ids), len(self.pairs)

    def unlabeled_pairs(self) -> list[IncidentPair]:
        labeled_ids = {l.pair_id for l in self.labels}
        return [p for p in self.pairs if p.pair_id not in labeled_ids]


# ---------------------------------------------------------------------------
# Scoring results
# ---------------------------------------------------------------------------

class CorrelationScore(BaseModel):
    """Model's correlation score for a pair of incidents."""

    incident_a_id: str
    incident_b_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class HumanVsModelReport(BaseModel):
    """Results of comparing model predictions to human labels."""

    total_pairs: int
    labeled_pairs: int
    labelers: list[str]

    # Model performance at various thresholds
    precision_at_50: float  # model confidence > 0.5 threshold
    recall_at_50: float
    f1_at_50: float

    precision_at_70: float
    recall_at_70: float
    f1_at_70: float

    precision_at_90: float
    recall_at_90: float
    f1_at_90: float

    # Calibration
    calibration_buckets: list[dict]  # [{conf_range, mean_conf, human_agree_rate, count}]

    # Agreement
    human_yes_rate: float  # fraction of pairs labeled "yes"
    human_maybe_rate: float
    inter_annotator_agreement: float | None = None  # Cohen's kappa if multiple labelers

    # Red herring detection
    high_conf_disagreements: int  # model confident but humans said "no"
    low_conf_agreements: int  # model not confident but humans said "yes"
