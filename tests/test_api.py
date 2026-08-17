from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from listing_to_reel.api.service import create_app


def test_fixture_job_reaches_terminal_state_and_exposes_artifacts(tmp_path: Path) -> None:
    sources = []
    for index, color in enumerate(["#224466", "#668844"]):
        path = tmp_path / f"source-{index}.jpg"
        Image.new("RGB", (320, 180), color).save(path)
        sources.append(path)
    app = create_app(tmp_path / "jobs.sqlite", tmp_path / "artifacts")
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        response = client.post(
            "/jobs", json={"kind": "fixture_reel", "source_paths": [str(path) for path in sources]}
        )
        assert response.status_code == 202
        job = client.get(f"/jobs/{response.json()['id']}").json()
        assert job["status"] == "succeeded"
        artifacts = client.get(f"/jobs/{job['id']}/artifacts").json()
        assert {item["name"] for item in artifacts} == {"manifest.json", "listing_reel.mp4"}
        assert client.get(f"/jobs/{job['id']}/artifacts/listing_reel.mp4").status_code == 200


def test_failed_job_can_be_retried(tmp_path: Path) -> None:
    app = create_app(tmp_path / "jobs.sqlite", tmp_path / "artifacts")
    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "kind": "fixture_reel",
                "source_paths": [str(tmp_path / "missing-a.jpg"), str(tmp_path / "missing-b.jpg")],
            },
        )
        job_id = response.json()["id"]
        assert client.get(f"/jobs/{job_id}").json()["status"] == "failed"
        retried = client.post(f"/jobs/{job_id}/retry")
        assert retried.status_code == 202
        assert client.get(f"/jobs/{job_id}").json()["status"] == "failed"
