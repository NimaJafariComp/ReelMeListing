from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from listing_to_reel.media.models import ReelRequest
from listing_to_reel.media.reel import assemble_reel


class SubmitJob(BaseModel):
    kind: str = "fixture_reel"
    source_paths: list[Path] = Field(min_length=2, max_length=12)


class Job(BaseModel):
    id: str
    kind: str
    status: str
    payload: dict[str, object]
    result: dict[str, object] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class Artifact(BaseModel):
    name: str
    path: str
    sha256: str
    size_bytes: int


class JobRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"
            )
            connection.execute("INSERT OR IGNORE INTO schema_migrations VALUES (1)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                "id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, "
                "payload TEXT NOT NULL, "
                "result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS artifacts (
                job_id TEXT NOT NULL, name TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL, PRIMARY KEY(job_id, name))""")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        data = dict(row)
        data["payload"] = json.loads(data["payload"])
        data["result"] = json.loads(data["result"]) if data["result"] else None
        return Job(**data)

    def create(self, request: SubmitJob) -> Job:
        now = datetime.now(UTC).isoformat()
        job = Job(
            id=uuid.uuid4().hex,
            kind=request.kind,
            status="queued",
            payload=request.model_dump(mode="json"),
            created_at=now,
            updated_at=now,
        )
        with self._connect() as c:
            c.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)",
                (job.id, job.kind, job.status, json.dumps(job.payload), now, now),
            )
        return job

    def get(self, job_id: str) -> Job | None:
        with self._connect() as c:
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def list(self) -> list[Job]:
        with self._connect() as c:
            rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._job(row) for row in rows]

    def claim(self) -> Job | None:
        with self._connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            now = datetime.now(UTC).isoformat()
            c.execute(
                "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?", (now, row["id"])
            )
            data = dict(row)
            data["status"] = "running"
            data["updated_at"] = now
            data["payload"] = json.loads(data["payload"])
            data["result"] = json.loads(data["result"]) if data["result"] else None
            return Job(**data)

    def finish(
        self, job_id: str, result: dict[str, object] | None, error: str | None = None
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as c:
            c.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = ? WHERE id = ?",
                (
                    "failed" if error else "succeeded",
                    json.dumps(result) if result else None,
                    error,
                    now,
                    job_id,
                ),
            )

    def retry(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None or job.status != "failed":
            return None
        now = datetime.now(UTC).isoformat()
        with self._connect() as c:
            c.execute(
                "UPDATE jobs SET status = 'queued', result = NULL, error = NULL, "
                "updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        return self.get(job_id)

    def add_artifact(self, job_id: str, artifact: Artifact) -> None:
        with self._connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?)",
                (job_id, artifact.name, artifact.path, artifact.sha256, artifact.size_bytes),
            )

    def artifacts(self, job_id: str) -> list[Artifact]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT name, path, sha256, size_bytes FROM artifacts WHERE job_id = ?", (job_id,)
            ).fetchall()
        return [Artifact(**dict(row)) for row in rows]


def _artifact(path: Path) -> Artifact:
    return Artifact(
        name=path.name,
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


class Worker:
    def __init__(self, repository: JobRepository, artifact_root: Path) -> None:
        self.repository, self.artifact_root = repository, artifact_root

    def run_once(self) -> Job | None:
        job = self.repository.claim()
        if not job:
            return None
        try:
            if job.kind != "fixture_reel":
                raise ValueError(f"Unsupported job kind: {job.kind}")
            paths = [Path(value) for value in job.payload["source_paths"]]
            if not all(path.is_file() for path in paths):
                raise FileNotFoundError("One or more fixture source images are missing.")
            work = self.artifact_root / job.id
            manifest = assemble_reel(ReelRequest(source_paths=paths, output_dir=work))
            result = {
                "run_id": manifest.run_id,
                "source_coverage": {path: "included" for path in manifest.source_paths},
            }
            for path in [Path(manifest.output_path), work / manifest.run_id / "manifest.json"]:
                self.repository.add_artifact(job.id, _artifact(path))
            self.repository.finish(job.id, result)
        except Exception as error:
            self.repository.finish(job.id, None, str(error))
        return self.repository.get(job.id)


def create_app(
    db_path: Path = Path("runs/service/jobs.sqlite"),
    artifact_root: Path = Path("runs/service/artifacts"),
) -> FastAPI:
    repository = JobRepository(db_path)
    worker = Worker(repository, artifact_root)
    app = FastAPI(title="ReelMeListing local job API", version="0.1.0")

    def require(job_id: str) -> Job:
        job = repository.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/jobs", response_model=Job, status_code=202)
    def submit(request: SubmitJob, tasks: BackgroundTasks) -> Job:
        job = repository.create(request)
        tasks.add_task(worker.run_once)
        return job

    @app.get("/jobs", response_model=list[Job])
    def jobs() -> list[Job]:
        return repository.list()

    @app.get("/jobs/{job_id}", response_model=Job)
    def job(job_id: str) -> Job:
        return require(job_id)

    @app.post("/jobs/{job_id}/retry", response_model=Job, status_code=202)
    def retry(job_id: str, tasks: BackgroundTasks) -> Job:
        if not repository.retry(job_id):
            raise HTTPException(409, "Only failed jobs can be retried")
        tasks.add_task(worker.run_once)
        return require(job_id)

    @app.get("/jobs/{job_id}/artifacts", response_model=list[Artifact])
    def artifacts(job_id: str) -> list[Artifact]:
        require(job_id)
        return repository.artifacts(job_id)

    @app.get("/jobs/{job_id}/artifacts/{name}")
    def download(job_id: str, name: str) -> FileResponse:
        artifact = next((item for item in repository.artifacts(job_id) if item.name == name), None)
        if artifact is None:
            raise HTTPException(404, "Artifact not found")
        return FileResponse(artifact.path, filename=name)

    return app
