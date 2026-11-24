"""FastAPI app: alert queue, alert detail, narrative drafts, analyst decisions, model metrics.

    uvicorn src.api.main:app --reload        # then open http://127.0.0.1:8000

The model ranks alerts and the LLM drafts text; only a human analyst records a decision.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api import db
from src.api.schemas import (
    AlertDetail, AlertPage, AlertSummary, DecisionOut, DecisionRequest, Factor, NarrativeOut,
    NarrativeRequest, Status,
)
from src.config import Settings, get_settings
from src.llm.facts import account_history, build_facts
from src.llm.generator import generate_narrative
from src.llm.providers import get_provider
from src.llm.schema import NarrativeValidation, SARNarrative, render_text

STATUS_OF = {"approve": "approved", "dismiss": "dismissed"}
DECISION_OF = {v: k for k, v in STATUS_OF.items()}
LATEST_DECISION = """LEFT JOIN (SELECT alert_id, decision FROM decisions
                                WHERE id IN (SELECT MAX(id) FROM decisions GROUP BY alert_id)) d
                     ON d.alert_id = a.alert_id"""


def _now() -> str:
    """Current UTC time, ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status(decision: str | None) -> Status:
    """Map the latest decision to a queue status."""
    return STATUS_OF.get(decision or "", "pending")  # type: ignore[return-value]


def _summary(row: sqlite3.Row) -> AlertSummary:
    """Queue row from a joined alerts/decisions row."""
    return AlertSummary(alert_id=row["alert_id"], rank=row["rank"], score=row["score"], ts=row["ts"],
                        src=row["src"], dst=row["dst"], amount=row["amount"], currency=row["pay_currency"],
                        payment_format=row["payment_format"], status=_status(row["decision"]))


def _latest_narrative(conn: sqlite3.Connection, alert_id: int) -> NarrativeOut | None:
    """Most recent narrative draft for an alert, if any."""
    r = conn.execute("SELECT * FROM narratives WHERE alert_id = ? ORDER BY id DESC LIMIT 1", (alert_id,)).fetchone()
    if r is None:
        return None
    narrative = SARNarrative.model_validate_json(r["payload_json"])
    return NarrativeOut(id=r["id"], alert_id=alert_id, created_at=r["created_at"], provider=r["provider"],
                        model=r["model"], narrative=narrative, text=render_text(narrative),
                        validation=NarrativeValidation.model_validate_json(r["validation_json"]))


def _decision_out(r: sqlite3.Row) -> DecisionOut:
    """API model from a decisions row."""
    return DecisionOut(id=r["id"], alert_id=r["alert_id"], decision=r["decision"], note=r["note"],
                       narrative_text=r["narrative_text"], created_at=r["created_at"], status=_status(r["decision"]))


def _latest_decision(conn: sqlite3.Connection, alert_id: int) -> DecisionOut | None:
    """Most recent analyst decision for an alert, if any."""
    r = conn.execute("SELECT * FROM decisions WHERE alert_id = ? ORDER BY id DESC LIMIT 1", (alert_id,)).fetchone()
    return _decision_out(r) if r else None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Settings are resolved per request unless given (tests pass their own)."""
    app = FastAPI(title="AML Transaction Monitoring", version="1.0")

    def cfg() -> Settings:
        return settings or get_settings()

    def get_conn() -> Iterator[sqlite3.Connection]:
        """One SQLite connection per request; 503 if the alert store hasn't been built."""
        s = cfg()
        if not s.db_path.exists():
            raise HTTPException(503, "Alert store not built yet. Run: python -m src.pipeline train")
        conn = db.connect(s.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def get_alert_row(alert_id: int, conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute(f"SELECT a.*, d.decision FROM alerts a {LATEST_DECISION} WHERE a.alert_id = ?",
                           (alert_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"Alert {alert_id} not found")
        return row

    @app.get("/api/alerts", response_model=AlertPage)
    def list_alerts(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                    min_score: float = Query(0.0, ge=0.0, le=1.0), status: Status | None = None,
                    conn: sqlite3.Connection = Depends(get_conn)) -> AlertPage:
        """Ranked alert queue (highest score first), filterable by status and minimum score."""
        where, params = ["a.score >= ?"], [min_score]
        if status == "pending":
            where.append("d.decision IS NULL")
        elif status:
            where.append("d.decision = ?")
            params.append(DECISION_OF[status])
        clause = " AND ".join(where)
        total = conn.execute(f"SELECT COUNT(*) FROM alerts a {LATEST_DECISION} WHERE {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT a.alert_id, a.rank, a.score, a.ts, a.src, a.dst, a.amount, a.pay_currency,
                       a.payment_format, d.decision
                FROM alerts a {LATEST_DECISION} WHERE {clause} ORDER BY a.rank LIMIT ? OFFSET ?""",
            [*params, limit, offset]).fetchall()
        return AlertPage(total=total, limit=limit, offset=offset, items=[_summary(r) for r in rows])

    @app.get("/api/alerts/{alert_id}", response_model=AlertDetail)
    def alert_detail(alert_id: int, conn: sqlite3.Connection = Depends(get_conn)) -> AlertDetail:
        """Alert details, SHAP factors, account summaries/history, latest narrative and decision."""
        row = get_alert_row(alert_id, conn)
        facts = build_facts(conn, alert_id)
        return AlertDetail(
            **_summary(row).model_dump(), src_bank=row["src_bank"], dst_bank=row["dst_bank"],
            pay_currency=row["pay_currency"], recv_currency=row["recv_currency"],
            factors=[Factor(**f) for f in json.loads(row["factors_json"])],
            features=json.loads(row["features_json"]),
            sender=facts.sender, receiver=facts.receiver,
            sender_history=account_history(conn, [row["src"]], row["ts"], limit=20),
            receiver_history=account_history(conn, [row["dst"]], row["ts"], limit=20),
            narrative=_latest_narrative(conn, alert_id), decision=_latest_decision(conn, alert_id),
        )

    @app.post("/api/alerts/{alert_id}/narrative", response_model=NarrativeOut)
    def create_narrative(alert_id: int, body: NarrativeRequest | None = None,
                         conn: sqlite3.Connection = Depends(get_conn)) -> NarrativeOut:
        """Generate (or regenerate) a validated narrative draft and store it."""
        get_alert_row(alert_id, conn)
        facts = build_facts(conn, alert_id)
        choice = body.provider if body else None
        if choice == "template":
            result = generate_narrative(facts, use_env=False)
        else:
            result = generate_narrative(facts, provider=get_provider(choice) if choice else None)
        created = _now()
        with conn:
            cur = conn.execute(
                "INSERT INTO narratives (alert_id, created_at, provider, model, payload_json, validation_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (alert_id, created, result.provider, result.model, result.narrative.model_dump_json(),
                 result.validation.model_dump_json()))
        return NarrativeOut(id=cur.lastrowid, alert_id=alert_id, created_at=created, provider=result.provider,
                            model=result.model, narrative=result.narrative, text=result.text,
                            validation=result.validation)

    @app.post("/api/alerts/{alert_id}/decision", response_model=DecisionOut)
    def record_decision(alert_id: int, body: DecisionRequest,
                        conn: sqlite3.Connection = Depends(get_conn)) -> DecisionOut:
        """Persist an analyst's approve/dismiss decision with a note (append-only audit trail)."""
        get_alert_row(alert_id, conn)
        with conn:
            cur = conn.execute(
                "INSERT INTO decisions (alert_id, decision, note, narrative_text, created_at) VALUES (?, ?, ?, ?, ?)",
                (alert_id, body.decision, body.note.strip(), body.narrative_text, _now()))
        return _decision_out(conn.execute("SELECT * FROM decisions WHERE id = ?", (cur.lastrowid,)).fetchone())

    @app.get("/api/metrics")
    def metrics(conn: sqlite3.Connection = Depends(get_conn)) -> dict[str, object]:
        """Model metrics (test split) plus review-queue counters for the dashboard."""
        s = cfg()
        path = s.model_dir / "metrics.json"
        out: dict[str, object] = {"available": path.exists()}
        if path.exists():
            m = json.loads(path.read_text())
            xg, iso = m["models"]["xgboost"], m["models"]["isolation_forest"]
            out["model"] = {
                "dataset": m["dataset"], "rows": m["rows"], "alert_budget": m["alert_budget"],
                "trained_at": m["created_at"], "use_graph": m["use_graph"],
                "xgboost": {k: xg["test"][k] for k in ("pr_auc", "recall_at_budget", "precision_at_budget",
                                                        "positive_rate")} | {"at_threshold": xg["test"]["at_threshold"]},
                "isolation_forest": {k: iso["test"][k] for k in ("pr_auc", "recall_at_budget")},
            }
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        counts = dict(conn.execute(f"""SELECT COALESCE(d.decision, 'pending'), COUNT(*)
                                       FROM alerts a {LATEST_DECISION} GROUP BY 1""").fetchall())
        out["queue"] = {
            "total": sum(counts.values()), "pending": counts.get("pending", 0),
            "approved": counts.get("approve", 0), "dismissed": counts.get("dismiss", 0),
            "reviewed_today": conn.execute("SELECT COUNT(DISTINCT alert_id) FROM decisions WHERE created_at LIKE ?",
                                           (today + "%",)).fetchone()[0],
        }
        return out

    frontend = (settings or get_settings()).frontend_dir
    app.mount("/static", StaticFiles(directory=frontend), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the single-page analyst UI."""
        return FileResponse(frontend / "index.html")

    return app


app = create_app()
