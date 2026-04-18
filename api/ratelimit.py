"""Per-tenant token-bucket rate limiter.

In-memory implementation suitable for single-process deployments and tests.
Production must replace ``_BUCKETS`` with Redis (or Memcached) so that
multiple API workers share the same allowance.
"""

from dataclasses import dataclass
import threading
import time

from fastapi import HTTPException, status

from api.tenants import Tenant


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


_BUCKETS: dict[str, _Bucket] = {}
_LOCK = threading.Lock()


def _refill(bucket: _Bucket, capacity: int, refill_per_sec: float, now: float) -> None:
    elapsed = max(0.0, now - bucket.last_refill)
    bucket.tokens = min(float(capacity), bucket.tokens + elapsed * refill_per_sec)
    bucket.last_refill = now


def consume(tenant: Tenant, cost: float = 1.0) -> None:
    """Spend ``cost`` tokens from ``tenant``'s bucket or raise 429.

    Capacity equals ``rate_per_min``; the bucket refills smoothly at
    ``rate_per_min / 60`` tokens per second.
    """
    capacity = tenant.rate_per_min
    refill_per_sec = capacity / 60.0
    now = time.monotonic()
    with _LOCK:
        bucket = _BUCKETS.get(tenant.tenant_id)
        if bucket is None:
            bucket = _Bucket(tokens=float(capacity), last_refill=now)
            _BUCKETS[tenant.tenant_id] = bucket
        _refill(bucket, capacity, refill_per_sec, now)
        if bucket.tokens < cost:
            retry_after = max(1, int((cost - bucket.tokens) / refill_per_sec))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.tokens -= cost


def reset_for_tests() -> None:
    with _LOCK:
        _BUCKETS.clear()
