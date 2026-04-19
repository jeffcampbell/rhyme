"""
SQLAlchemy database models for the web labeling tool.

Supports PostgreSQL (production) and SQLite (local development).
Set DATABASE_URL env var to configure:
  - PostgreSQL: postgresql://user:pass@host:5432/dbname
  - SQLite:     sqlite:///data/web/rhyme.db  (default)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

DEFAULT_DATABASE_URL = "sqlite:///data/web/rhyme.db"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


class DBSession(db.Model):
    """A labeling session — one per imported corpus."""

    __tablename__ = "sessions"

    id = db.Column(db.String(32), primary_key=True)
    name = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    incident_count = db.Column(db.Integer, default=0)
    pair_count = db.Column(db.Integer, default=0)

    incidents = db.relationship("DBIncident", backref="session", lazy="dynamic", cascade="all, delete-orphan")
    pairs = db.relationship("DBPair", backref="session", lazy="dynamic", cascade="all, delete-orphan")


class DBIncident(db.Model):
    """An imported incident."""

    __tablename__ = "incidents"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(32), db.ForeignKey("sessions.id"), nullable=False, index=True)
    incident_id = db.Column(db.String(255), nullable=False)  # user-provided ID (e.g., INC-001)
    summary = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.String(64), nullable=False)
    severity = db.Column(db.String(32), nullable=True)
    service = db.Column(db.String(255), nullable=True)
    url = db.Column(db.Text, nullable=True)  # link to incident in company's tracker
    metadata_json = db.Column(db.Text, default="{}")

    __table_args__ = (
        db.UniqueConstraint("session_id", "incident_id", name="uq_session_incident"),
    )


class DBPair(db.Model):
    """An incident pair for labeling."""

    __tablename__ = "pairs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pair_id = db.Column(db.String(32), nullable=False, unique=True, index=True)
    session_id = db.Column(db.String(32), db.ForeignKey("sessions.id"), nullable=False, index=True)
    incident_a_id = db.Column(db.String(255), nullable=False)
    incident_b_id = db.Column(db.String(255), nullable=False)
    model_confidence = db.Column(db.Float, nullable=True)
    sampling_bucket = db.Column(db.String(32), nullable=False, default="random")

    labels = db.relationship("DBLabel", backref="pair", lazy="dynamic", cascade="all, delete-orphan")


class DBLabel(db.Model):
    """A human label for an incident pair."""

    __tablename__ = "labels"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pair_id = db.Column(db.String(32), db.ForeignKey("pairs.pair_id"), nullable=False, index=True)
    session_id = db.Column(db.String(32), db.ForeignKey("sessions.id"), nullable=False, index=True)
    labeler_id = db.Column(db.String(255), nullable=False, default="default")
    judgment = db.Column(db.String(16), nullable=False)  # yes, no, maybe
    notes = db.Column(db.Text, default="")
    labeled_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


def init_db(app):
    """Initialize database with the Flask app."""
    database_url = get_database_url()
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
