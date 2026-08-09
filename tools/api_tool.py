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
    """Send an HTTP request to a local app; returns status, headers, body.

    Args:
        url: Full URL, e.g. "http://localhost:8000/users/1".
        method: GET/POST/PUT/PATCH/DELETE. Default GET.
        json_body: Optional JSON dict request body.
        headers: Optional request headers.
        timeout: Max seconds to wait.
    """

    try:
        response = httpx.request(
            method=method.upper(),
            url=url,
            json=json_body,
            headers=headers,
            timeout=timeout
        )
    except httpx.RequestError as e:
        return f"ERROR: request failed: {e}"

    try:
        body = response.text[:4000]
    except Exception:
        body = "(ubable to decode body)"

    return f"STATUS: {response.status_code}\nHEADERS: {dict(response.headers)}\nBODY:\n{body}"


@tool
def fetch_openapi_schema(base_url: str) -> str:
    """Fetch a FastAPI app's OpenAPI schema (routes/params/models) instead
    of reading every route file.

    Args:
        base_url: App base URL, e.g. "http://localhost:8000".
    """
    url = base_url.rstrip("/") + "/openapi.json"
    try:
        response = httpx.get(url, timeout=10)
    except httpx.RequestError as e:
        return f"ERROR: request failed: {e}"
    
    if response.status_code != 200:
        return f"ERROR: status {response.status_code} fetching {url}"
    
    return response.text[:12000]
