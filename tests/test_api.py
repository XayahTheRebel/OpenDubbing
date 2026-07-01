import pytest
from fastapi.testclient import TestClient

from opendubbing.api.server import create_app
from opendubbing.config import AppConfig


@pytest.fixture
def client():
    config = AppConfig()
    app = create_app(config)
    return TestClient(app)


class TestAPI:
    def test_health(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_list_providers(self, client):
        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        providers = response.json()
        kinds = {p["kind"] for p in providers}
        assert "tts" in kinds
        assert "asr" in kinds

    def test_create_task(self, client, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "workspace:\n  root: ./workspace\n"
            "pipeline:\n  steps:\n    - input\n"
        )
        response = client.post(
            "/api/v1/tasks",
            json={
                "input_path": str(tmp_path / "video.mp4"),
                "config_path": str(config_path),
                "resume": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] in {"pending", "running"}

    def test_get_task_not_found(self, client):
        response = client.get("/api/v1/tasks/nonexistent")
        assert response.status_code == 404
