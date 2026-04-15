from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_synthesize_contract():
    response = client.post(
        "/synthesize",
        json={"text": "Mhoro", "language": "shona", "voice": "female_1", "speed": 1.0},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")


def test_synthesize_rejects_unknown_language():
    response = client.post(
        "/synthesize",
        json={"text": "hello", "language": "klingon", "voice": "female_1", "speed": 1.0},
    )
    assert response.status_code == 400
