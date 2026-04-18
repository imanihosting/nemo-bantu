"""API key authentication.

FastAPI dependency that validates the ``X-API-Key`` header against the
tenant store and returns the resolved :class:`Tenant`. Wire into routes
that require auth via ``Depends(require_tenant)``.
"""

from fastapi import Header, HTTPException, status

from api.tenants import Tenant, get_store


API_KEY_HEADER = "X-API-Key"


def require_tenant(x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> Tenant:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"missing {API_KEY_HEADER} header",
        )
    tenant = get_store().lookup(x_api_key)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
    return tenant
