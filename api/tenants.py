"""In-memory tenant + API key store.

Bootstrap implementation for Phase D commercial hardening. Production must
swap this for a Postgres-backed store with key rotation and revocation; the
Tenant/TenantStore interfaces are designed so the swap is local to this file.

Configuration:
    API_KEYS env var, comma-separated ``key:tenant_id[:rate_per_min]`` triples.
    Example:  ``API_KEYS=abc123:acme:120,xyz789:globex:30``

Dev mode:
    If API_KEYS is unset, the store auto-provisions a single tenant ``dev``
    with key ``dev-key`` and a generous rate limit. This keeps local
    development and the existing test suite working without per-test setup.
"""

from dataclasses import dataclass
import os
import threading


DEFAULT_RATE_PER_MIN = 60
DEV_API_KEY = "dev-key"


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    rate_per_min: int


class TenantStore:
    def __init__(self, env_value: str | None = None) -> None:
        self._lock = threading.Lock()
        self._by_key: dict[str, Tenant] = {}
        self._load(env_value if env_value is not None else os.environ.get("API_KEYS"))

    def _load(self, env_value: str | None) -> None:
        if not env_value:
            self._by_key[DEV_API_KEY] = Tenant(tenant_id="dev", rate_per_min=600)
            return
        for raw in env_value.split(","):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(":")
            if len(parts) < 2:
                continue
            key = parts[0]
            tenant_id = parts[1]
            rate = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else DEFAULT_RATE_PER_MIN
            self._by_key[key] = Tenant(tenant_id=tenant_id, rate_per_min=rate)

    def lookup(self, api_key: str) -> Tenant | None:
        with self._lock:
            return self._by_key.get(api_key)


_store: TenantStore | None = None


def get_store() -> TenantStore:
    global _store
    if _store is None:
        _store = TenantStore()
    return _store


def reset_store_for_tests(env_value: str | None = None) -> None:
    """Test hook: rebuild the store with a custom env string. Production code
    must not call this."""
    global _store
    _store = TenantStore(env_value=env_value)
