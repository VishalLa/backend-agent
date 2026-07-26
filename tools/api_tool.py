import httpx
from langchain_core.tools import tool


@tool
def http_request(
    url: str,
    method: str = "GET",
    json_body: dict = None,
    headers: dict = None,
    timeout: int = 15,
) -> str:
    """Send an HTTP request to a locally running Flask/FastAPI app and return
    the status code, headers, and body. Use this to smoke-test endpoints
    after making changes, instead of writing throwaway curl commands.

    Args:
        url: Full URL to request, e.g. "http://localhost:8000/users/1".
        method: HTTP method: GET, POST, PUT, PATCH, or DELETE. Defaults to GET.
        json_body: Optional JSON-serializable dict to send as the request body.
        headers: Optional dict of request headers.
        timeout: Max seconds to wait for a response.
    """
    try:
        response = httpx.request(
            method.upper(), url, json=json_body, headers=headers, timeout=timeout
        )
    except httpx.RequestError as e:
        return f"ERROR: request failed: {e}"

    try:
        body = response.text[:4000]
    except Exception:  # noqa: BLE001
        body = "(unable to decode body)"

    return f"STATUS: {response.status_code}\nHEADERS: {dict(response.headers)}\nBODY:\n{body}"


@tool
def fetch_openapi_schema(base_url: str) -> str:
    """Fetch a FastAPI app's auto-generated OpenAPI schema (openapi.json) so
    the agent can see the full API surface (routes, params, models) without
    re-reading every route file.

    Args:
        base_url: Base URL of the running app, e.g. "http://localhost:8000".
    """
    url = base_url.rstrip("/") + "/openapi.json"
    try:
        response = httpx.get(url, timeout=10)
    except httpx.RequestError as e:
        return f"ERROR: request failed: {e}"
    if response.status_code != 200:
        return f"ERROR: status {response.status_code} fetching {url}"
    return response.text[:12000]
    