from typing import Any, Dict

import httpx


async def make_request(url: str, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """HTTP client adapter for external weather API calls.

    Provides a thin wrapper around httpx for dependency injection.
    Handles common HTTP concerns like timeouts and error responses.

    Args:
        url: Target URL for GET request
        headers: Optional HTTP headers dict

    Returns:
        Parsed JSON response as dictionary, or None if request failed

    Note:
        Returns None on any HTTP or parsing error to simplify
        error handling in domain services. Consider logging errors
        at this layer for debugging.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None
