import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.ratelimit import reset_for_tests
from api.tenants import DEV_API_KEY, reset_store_for_tests


@pytest.fixture(autouse=True)
def _reset_state():
    # Each test gets a fresh tenant store and rate-limit buckets so test
    # ordering cannot affect 429 outcomes.
    reset_store_for_tests(env_value=None)  # dev-mode tenant
    reset_for_tests()
    yield
    reset_for_tests()


client = TestClient(app)
HEADERS = {"X-API-Key": DEV_API_KEY}


def test_health_does_not_require_auth():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_synthesize_contract():
    response = client.post(
        "/synthesize",
        json={"text": "Mhoro", "language": "shona", "voice": "female_1", "speed": 1.0},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")


def test_synthesize_rejects_unknown_language():
    response = client.post(
        "/synthesize",
        json={"text": "hello", "language": "klingon", "voice": "female_1", "speed": 1.0},
        headers=HEADERS,
    )
    assert response.status_code == 400


def test_synthesize_requires_api_key():
    response = client.post(
        "/synthesize",
        json={"text": "Mhoro", "language": "shona", "voice": "female_1", "speed": 1.0},
    )
    assert response.status_code == 401


def test_synthesize_rejects_invalid_api_key():
    response = client.post(
        "/synthesize",
        json={"text": "Mhoro", "language": "shona", "voice": "female_1", "speed": 1.0},
        headers={"X-API-Key": "not-a-real-key"},
    )
    assert response.status_code == 401


def test_voices_endpoint_lists_one_per_language():
    response = client.get("/voices", headers=HEADERS)
    assert response.status_code == 200
    voices = response.json()
    languages = {v["language"] for v in voices}
    assert languages == {"shona", "ndebele", "zulu", "xhosa"}


def test_response_carries_request_id_header():
    response = client.get("/health")
    assert "x-request-id" in {k.lower() for k in response.headers.keys()}


def test_rate_limit_returns_429_after_burst():
    # Provision a tiny per-minute budget and exhaust it. Confirms the bucket
    # actually denies; without this test, mis-wired auth could mask the limiter.
    reset_store_for_tests(env_value=f"tinykey:tiny:1")  # 1 req/min
    headers = {"X-API-Key": "tinykey"}
    first = client.post(
        "/synthesize",
        json={"text": "Mhoro", "language": "shona", "voice": "female_1", "speed": 1.0},
        headers=headers,
    )
    assert first.status_code == 200
    second = client.post(
        "/synthesize",
        json={"text": "Mhoro", "language": "shona", "voice": "female_1", "speed": 1.0},
        headers=headers,
    )
    assert second.status_code == 429
    assert "retry-after" in {k.lower() for k in second.headers.keys()}
