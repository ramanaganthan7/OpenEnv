"""
ClimateWatch — Server Entry Point
==================================
ASGI wrapper that handles empty POST bodies for /reset and /step.

Usage:
  uvicorn serve:app --host 0.0.0.0 --port 7860

Why this file exists:
  The hackathon validator sends `POST /reset` with NO body.
  FastAPI returns 422 for empty bodies even with `Body(default=None)`.
  This ASGI wrapper intercepts empty POST bodies and injects `{}`
  before FastAPI sees the request.
"""

from app.main import app as fastapi_app

_INJECT_ROUTES = {"/reset", "/step"}


class _InjectEmptyBody:
    """ASGI wrapper: injects {} body for empty POST /reset and /step."""

    def __init__(self, inner_app):
        self._app = inner_app

    async def __call__(self, scope, receive, send):
        if (scope.get("type") == "http"
                and scope.get("method") == "POST"
                and scope.get("path") in _INJECT_ROUTES):

            # Read original body
            chunks = []
            more = True
            while more:
                msg = await receive()
                chunks.append(msg.get("body", b""))
                more = msg.get("more_body", False)
            raw = b"".join(chunks)

            # If empty, inject {}
            if not raw or not raw.strip():
                raw = b"{}"

            # Ensure Content-Type: application/json is present
            headers = list(scope.get("headers", []))
            has_ct = any(k.lower() == b"content-type" for k, v in headers)
            if not has_ct:
                headers = headers + [(b"content-type", b"application/json")]

            new_scope = dict(scope, headers=headers)

            body_sent = False
            async def patched_receive():
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {"type": "http.request", "body": raw, "more_body": False}
                return {"type": "http.disconnect"}

            await self._app(new_scope, patched_receive, send)
        else:
            await self._app(scope, receive, send)


# This is what uvicorn loads: serve:app
app = _InjectEmptyBody(fastapi_app)
