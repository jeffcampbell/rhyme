"""
Flask web application for incident correlation labeling and dashboard.

Routes:
  /                     — Home: import corpus or resume session
  /import               — Upload incident JSON
  /session/<id>         — Labeling interface
  /session/<id>/label   — Submit a label (POST)
  /dashboard/<id>       — Results dashboard
  /health               — Health check for load balancers
  /api/session/<id>     — Session data as JSON
  /api/export/<id>      — Export results as JSON
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash

from .database import db, init_db, DBSession, DBIncident, DBPair, DBLabel, DBModelScore
from .models import (
    OrgCorpus,
    OrgIncident,
    IncidentPair,
    LabelingSession,
    PairLabel,
    parse_incidents,
    ImportResult,
)
from .pair_sampler import sample_pairs
from .scorer_human import score_against_humans, score_model_against_humans

logger = logging.getLogger("rhyme_web")


def create_app(data_dir: str | None = None, verbose: bool = False) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("SECRET_KEY", uuid.uuid4().hex)

    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(level)

    # For SQLite, use data_dir to set the path
    if data_dir and "DATABASE_URL" not in os.environ:
        from pathlib import Path
        db_dir = Path(data_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("DATABASE_URL", f"sqlite:///{db_dir.resolve()}/rhyme.db")

    init_db(app)
    logger.info(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/about")
    def about():
        return render_template("about.html")

    # ------------------------------------------------------------------
    # Home
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        sessions = []
        for s in DBSession.query.order_by(DBSession.created_at.desc()).all():
            labeled = db.session.query(db.func.count(db.func.distinct(DBLabel.pair_id))).filter(DBLabel.session_id == s.id).scalar()
            total = s.pair_count
            sessions.append({
                "id": s.id,
                "name": s.name,
                "created": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "",
                "pairs": total,
                "labeled": labeled,
                "pct": int(100 * labeled / total) if total else 0,
            })
        return render_template("index.html", sessions=sessions)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    @app.route("/import", methods=["GET", "POST"])
    def import_page():
        if request.method == "GET":
            return render_template("import.html")

        json_text = None
        if "file" in request.files and request.files["file"].filename:
            json_text = request.files["file"].read().decode("utf-8")
        elif request.form.get("json_text"):
            json_text = request.form["json_text"]

        if not json_text:
            logger.warning("Import attempted with no data")
            flash("No data provided", "error")
            return render_template("import.html")

        try:
            result = parse_incidents(json_text)
            corpus = result.corpus
            logger.info(f"Parsed {len(corpus.incidents)} incidents from import")
            if result.has_errors:
                for err in result.errors:
                    logger.warning(f"Import warning: {err}")
        except Exception as e:
            logger.error(f"Import parse error: {e}")
            flash(f"Import error: {e}", "error")
            return render_template("import.html")

        if result.has_errors and len(corpus.incidents) > 0:
            error_summary = f"{len(result.errors)} incident(s) skipped due to errors"
            if len(result.errors) <= 5:
                error_detail = "; ".join(result.errors)
                flash(f"{error_summary}: {error_detail}", "error")
            else:
                first_few = "; ".join(result.errors[:3])
                flash(f"{error_summary} (showing first 3): {first_few}", "error")

        if len(corpus.incidents) < 5:
            msg = f"Only {len(corpus.incidents)} valid incident(s) — need at least 5"
            if result.has_errors:
                msg += f" ({len(result.errors)} skipped due to errors)"
            logger.warning(msg)
            flash(msg, "error")
            return render_template("import.html")

        # Create pair sampling
        total_pairs = min(200, len(corpus.incidents) * (len(corpus.incidents) - 1) // 2)
        total_pairs = max(10, total_pairs)
        labeling_session = sample_pairs(corpus, total_pairs=total_pairs)

        # Save to database
        session_id = labeling_session.session_id
        session_name = request.form.get("session_name", "").strip() or None
        db_session = DBSession(
            id=session_id,
            name=session_name,
            incident_count=len(corpus.incidents),
            pair_count=len(labeling_session.pairs),
        )
        db.session.add(db_session)

        for inc in corpus.incidents:
            db.session.add(DBIncident(
                session_id=session_id,
                incident_id=inc.id,
                summary=inc.summary,
                timestamp=inc.timestamp,
                severity=inc.severity,
                service=inc.service,
                url=inc.url,
            ))

        for pair in labeling_session.pairs:
            db.session.add(DBPair(
                pair_id=pair.pair_id,
                session_id=session_id,
                incident_a_id=pair.incident_a_id,
                incident_b_id=pair.incident_b_id,
                model_confidence=pair.model_confidence,
                sampling_bucket=pair.sampling_bucket,
            ))

        db.session.commit()
        logger.info(f"Created session {session_id}: {len(corpus.incidents)} incidents, {len(labeling_session.pairs)} pairs")
        flash(f"Imported {len(corpus.incidents)} incidents, created {len(labeling_session.pairs)} pairs for labeling", "success")
        return redirect(url_for("label_page", session_id=session_id))

    # ------------------------------------------------------------------
    # Labeling
    # ------------------------------------------------------------------

    @app.route("/session/<session_id>")
    def label_page(session_id: str):
        db_session = db.session.get(DBSession, session_id)
        if not db_session:
            logger.error(f"Session not found: {session_id}")
            flash("Session not found", "error")
            return redirect(url_for("index"))

        # Find unlabeled pairs
        labeled_pair_ids = {
            row[0] for row in
            db.session.query(DBLabel.pair_id).filter_by(session_id=session_id).distinct().all()
        }
        unlabeled = DBPair.query.filter(
            DBPair.session_id == session_id,
            ~DBPair.pair_id.in_(labeled_pair_ids) if labeled_pair_ids else True,
        ).all()

        if not unlabeled:
            return redirect(url_for("dashboard", session_id=session_id))

        pair = unlabeled[0]
        labeled_count = len(labeled_pair_ids)
        total = db_session.pair_count

        # Look up incidents
        inc_map = {
            inc.incident_id: inc
            for inc in DBIncident.query.filter_by(session_id=session_id).all()
        }
        inc_a = inc_map.get(pair.incident_a_id)
        inc_b = inc_map.get(pair.incident_b_id)

        return render_template(
            "label.html",
            session_id=session_id,
            session_name=db_session.name,
            pair=pair,
            inc_a=inc_a,
            inc_b=inc_b,
            labeled=labeled_count,
            total=total,
            pct=int(100 * labeled_count / total) if total else 0,
            remaining=len(unlabeled),
        )

    @app.route("/session/<session_id>/label", methods=["POST"])
    def submit_label(session_id: str):
        pair_id = request.form.get("pair_id", "")
        judgment = request.form.get("judgment", "")
        notes = request.form.get("notes", "")[:2000]
        labeler_id = request.form.get("labeler_id", "default")[:255]

        if judgment not in ("yes", "no", "maybe"):
            flash("Invalid judgment value", "error")
            return redirect(url_for("label_page", session_id=session_id))

        if not pair_id:
            flash("Missing pair ID", "error")
            return redirect(url_for("label_page", session_id=session_id))

        db.session.add(DBLabel(
            pair_id=pair_id,
            session_id=session_id,
            labeler_id=labeler_id,
            judgment=judgment,
            notes=notes,
        ))
        db.session.commit()

        labeled_count = db.session.query(db.func.count(db.func.distinct(DBLabel.pair_id))).filter(DBLabel.session_id == session_id).scalar()
        total = db.session.get(DBSession, session_id).pair_count
        logger.info(f"Label: session={session_id} pair={pair_id} judgment={judgment} ({labeled_count}/{total})")

        return redirect(url_for("label_page", session_id=session_id))

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    @app.route("/dashboard/<session_id>")
    def dashboard(session_id: str):
        db_session = db.session.get(DBSession, session_id)
        if not db_session:
            logger.error(f"Session not found: {session_id}")
            flash("Session not found", "error")
            return redirect(url_for("index"))

        labeled_count = db.session.query(db.func.count(db.func.distinct(DBLabel.pair_id))).filter(DBLabel.session_id == session_id).scalar()
        total = db_session.pair_count

        model_reports = {}
        report = None
        best_model = None

        if labeled_count > 0:
            ls = _build_labeling_session(session_id)

            # Score each uploaded model
            model_names = [
                r[0] for r in db.session.query(DBModelScore.model_name)
                .filter_by(session_id=session_id).distinct().all()
            ]
            for model_name in model_names:
                scores = DBModelScore.query.filter_by(
                    session_id=session_id, model_name=model_name
                ).all()
                conf_map = {s.pair_id: s.confidence for s in scores}
                model_reports[model_name] = score_model_against_humans(ls, conf_map)

            # Pick best model by F1 at 0.5
            if model_reports:
                best_model = max(model_reports, key=lambda m: model_reports[m].f1_at_50)
                report = model_reports[best_model]
            else:
                # Fallback to import-time model_confidence if no models uploaded
                report = score_against_humans(ls)

        return render_template(
            "dashboard.html",
            session_id=session_id,
            session=db_session,
            labeled=labeled_count,
            total=total,
            report=report,
            model_reports=model_reports,
            best_model=best_model,
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @app.route("/api/session/<session_id>")
    def api_session(session_id: str):
        ls = _build_labeling_session(session_id)
        if ls is None:
            return jsonify({"error": "Session not found"}), 404
        return jsonify(ls.model_dump())

    @app.route("/api/export/<session_id>")
    def api_export(session_id: str):
        ls = _build_labeling_session(session_id)
        if ls is None:
            return jsonify({"error": "Session not found"}), 404
        labeled_count = len(ls.labels)
        report = score_against_humans(ls) if labeled_count > 0 else None
        incidents = [
            {"id": inc.incident_id, "summary": inc.summary, "timestamp": inc.timestamp,
             "severity": inc.severity, "service": inc.service}
            for inc in DBIncident.query.filter_by(session_id=session_id).all()
        ]
        return jsonify({
            "session": ls.model_dump(),
            "incidents": incidents,
            "report": report.model_dump() if report else None,
        })

    @app.route("/api/scores/<session_id>", methods=["POST"])
    def api_upload_scores(session_id: str):
        """Upload model scores for a session's pairs."""
        db_session_obj = db.session.get(DBSession, session_id)
        if not db_session_obj:
            return jsonify({"error": "Session not found"}), 404

        data = request.get_json()
        if not data or "model_name" not in data or "scores" not in data:
            return jsonify({"error": "Must provide model_name and scores"}), 400

        model_name = data["model_name"][:255]
        scores = data["scores"]

        # Validate pair_ids belong to this session
        valid_pair_ids = {
            p.pair_id for p in DBPair.query.filter_by(session_id=session_id).all()
        }

        # Delete existing scores for this model+session (upsert semantics)
        DBModelScore.query.filter_by(
            session_id=session_id, model_name=model_name
        ).delete()

        count = 0
        for s in scores:
            pair_id = s.get("pair_id", "")
            if pair_id not in valid_pair_ids:
                continue
            try:
                conf = max(0.0, min(1.0, float(s.get("confidence", 0.0))))
            except (TypeError, ValueError):
                continue
            db.session.add(DBModelScore(
                session_id=session_id,
                model_name=model_name,
                pair_id=pair_id,
                confidence=conf,
            ))
            count += 1

        db.session.commit()
        logger.info(f"Uploaded {count} scores for model '{model_name}' in session {session_id}")
        return jsonify({"saved": count, "model_name": model_name})

    @app.route("/api/models/<session_id>")
    def api_list_models(session_id: str):
        """List models with uploaded scores for a session."""
        rows = db.session.query(
            DBModelScore.model_name,
            db.func.count(DBModelScore.id),
        ).filter_by(session_id=session_id).group_by(DBModelScore.model_name).all()
        return jsonify({"models": [{"name": r[0], "score_count": r[1]} for r in rows]})

    def _build_labeling_session(session_id: str) -> LabelingSession | None:
        """Reconstruct a LabelingSession from the database for scoring."""
        db_session = db.session.get(DBSession, session_id)
        if not db_session:
            return None

        pairs = [
            IncidentPair(
                pair_id=p.pair_id,
                incident_a_id=p.incident_a_id,
                incident_b_id=p.incident_b_id,
                model_confidence=p.model_confidence,
                sampling_bucket=p.sampling_bucket,
            )
            for p in DBPair.query.filter_by(session_id=session_id).all()
        ]

        labels = [
            PairLabel(
                pair_id=l.pair_id,
                labeler_id=l.labeler_id,
                judgment=l.judgment,
                notes=l.notes or "",
                labeled_at=l.labeled_at.isoformat() if l.labeled_at else "",
            )
            for l in DBLabel.query.filter_by(session_id=session_id).all()
        ]

        return LabelingSession(
            session_id=session_id,
            corpus_path="",
            pairs=pairs,
            labels=labels,
            created_at=db_session.created_at.isoformat() if db_session.created_at else "",
        )

    return app
