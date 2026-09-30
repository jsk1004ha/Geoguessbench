"""Separate, persistent evaluator API. Run one Uvicorn worker; labels stay server-side."""
import hashlib
import importlib.metadata
import json
import platform
import random
import secrets
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import Field

from . import PROTOCOL_VERSION, __version__
from .dataset import digest_json
from .engine import Episode
from .metrics import summarize, telemetry_summary
from .schema import Strict


class StartRequest(Strict):
    model: str = Field(min_length=1, max_length=160)
    settings: dict = Field(default_factory=dict)


class StepRequest(Strict):
    episode: str = Field(pattern=r"^[0-9a-f]{32}$")
    step: int = Field(ge=0)
    action: dict
    telemetry: dict = Field(default_factory=dict)


class Evaluator:
    def __init__(self, dataset, protocol, db_path):
        dataset.validate_track(protocol.track, protocol.split)
        self.dataset, self.protocol = dataset, protocol
        source = {path.name: hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
                  for path in sorted(Path(__file__).parent.glob("*.py"))}
        self.runtime = {"python": platform.python_version(), "platform": platform.system(),
                        "implementation_sha256": digest_json(source),
                        "dependencies": {name: importlib.metadata.version(name)
                                         for name in ("numpy", "Pillow", "pydantic", "fastapi", "httpx")}}
        self.protocol_hash = digest_json({"version": PROTOCOL_VERSION, "config": protocol.model_dump(),
                                         "implementation_sha256": self.runtime["implementation_sha256"]})
        self.lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)")
        self.db.commit()

    def close(self):
        self.db.close()

    def load(self, run_id):
        row = self.db.execute("SELECT snapshot FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError("unknown run")
        run = json.loads(row[0])
        if (run["dataset_sha256"] != self.dataset.fingerprint or run["protocol_sha256"] != self.protocol_hash):
            raise ValueError("run belongs to a different dataset or protocol")
        return run

    def save(self, run):
        self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?)", (run["id"], json.dumps(run, allow_nan=False)))
        self.db.commit()

    def create(self, model, settings):
        if len(json.dumps(settings)) > 16000:
            raise ValueError("settings too large")
        rounds = sorted(r.id for r in self.dataset.rounds.values() if r.split == self.protocol.split)
        random.Random(self.protocol.seed).shuffle(rounds)
        run = {"id": uuid.uuid4().hex, "model": model, "settings": settings,
               "dataset_sha256": self.dataset.fingerprint, "protocol_sha256": self.protocol_hash,
               "protocol": self.protocol.model_dump(), "protocol_version": PROTOCOL_VERSION,
               "package_version": __version__, "runtime": self.runtime, "fixture": self.dataset.manifest.fixture,
               "order": rounds, "cursor": 0, "active": None, "rows": [], "traces": []}
        self.save(run)
        return {"run_id": run["id"], "rounds": len(rounds), "dataset_sha256": self.dataset.fingerprint,
                "protocol_sha256": self.protocol_hash}

    def _collect(self, run, episode):
        if episode.state["status"] != "active":
            run["rows"].append(episode.state["result"])
            run["traces"].append(episode.state)
            run["active"] = None
            run["cursor"] += 1

    def next(self, run_id):
        run = self.load(run_id)
        if run["active"]:
            episode = Episode(self.dataset, self.protocol, run["active"]["round_id"], state=run["active"])
            episode.expire()
            self._collect(run, episode)
        if run["cursor"] == len(run["order"]):
            self.save(run)
            return {"complete": True, "run_id": run_id, "completed": run["cursor"]}
        if run["active"] is None:
            episode = Episode(self.dataset, self.protocol, run["order"][run["cursor"]])
            run["active"] = episode.state
        else:
            episode = Episode(self.dataset, self.protocol, run["active"]["round_id"], state=run["active"])
        observation = episode.observation()
        self._collect(run, episode)
        self.save(run)
        return {"complete": False, "round_index": run["cursor"], "total": len(run["order"]),
                "observation": observation}

    def step(self, run_id, request):
        run = self.load(run_id)
        if run["active"] is None:
            raise ValueError("no active episode; request next first")
        if request.episode != run["active"]["episode"]:
            raise ValueError("stale episode; fetch the current observation")
        if len(json.dumps(request.action)) > 16000 or len(json.dumps(request.telemetry)) > 16000:
            raise ValueError("payload too large")
        episode = Episode(self.dataset, self.protocol, run["active"]["round_id"], state=run["active"])
        # Persist timeouts even when a late action is rejected; no restart grants extra time.
        episode.expire()
        if episode.state["status"] == "active":
            try:
                json.dumps(request.action, allow_nan=False)
                action = request.action
            except ValueError:
                action = {"type": "invalid"}
            episode.step(action, request.step)
            if episode.state["events"]:
                episode.state["events"][-1]["telemetry_self_reported"] = request.telemetry
        observation = episode.observation()
        self._collect(run, episode)
        self.save(run)
        return {"observation": observation, "completed": run["cursor"], "total": len(run["order"])}

    def report(self, run_id, private=False):
        run = self.load(run_id)
        if run["cursor"] != len(run["order"]):
            raise ValueError("run is incomplete; interim scores and labels are withheld")
        result = {key: run[key] for key in ("id", "model", "settings", "dataset_sha256", "protocol_sha256",
                                           "protocol", "protocol_version", "package_version", "runtime", "fixture")}
        result["metrics"] = summarize(run["rows"], samples=self.protocol.bootstrap_samples, seed=self.protocol.seed)
        result["usage"] = telemetry_summary(run["traces"])
        if private:
            result.update(rows=run["rows"], traces=run["traces"])
        return result


def create_app(dataset, protocol, db_path, token=""):
    evaluator = Evaluator(dataset, protocol, db_path)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            evaluator.close()

    app = FastAPI(title="Geoguessbench evaluator", version=__version__, docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.evaluator = evaluator
    static = Path(__file__).parent / "web"

    def authorize(authorization: str | None = Header(default=None)):
        if token and not secrets.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "authentication required")

    def invoke(method, *args):
        with evaluator.lock:
            try:
                return method(*args)
            except KeyError as exc:
                raise HTTPException(404, "run not found") from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/app.js")
    def js():
        return FileResponse(static / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def css():
        return FileResponse(static / "style.css", media_type="text/css")

    @app.get("/api/config", dependencies=[Depends(authorize)])
    def config():
        return {"protocol": protocol.model_dump(), "dataset_sha256": dataset.fingerprint,
                "protocol_sha256": evaluator.protocol_hash,
                "rounds": sum(r.split == protocol.split for r in dataset.rounds.values()),
                "fixture": dataset.manifest.fixture}

    @app.post("/api/runs", dependencies=[Depends(authorize)])
    def start(body: StartRequest):
        return invoke(evaluator.create, body.model, body.settings)

    @app.post("/api/runs/{run_id}/next", dependencies=[Depends(authorize)])
    def next_round(run_id: str):
        return invoke(evaluator.next, run_id)

    @app.post("/api/runs/{run_id}/step", dependencies=[Depends(authorize)])
    def step(run_id: str, body: StepRequest):
        return invoke(evaluator.step, run_id, body)

    @app.get("/api/runs/{run_id}/report", dependencies=[Depends(authorize)])
    def report(run_id: str):
        return invoke(evaluator.report, run_id)

    return app
