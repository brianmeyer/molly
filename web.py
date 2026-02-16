"""Web UI backend for Molly (Phase 4).

FastAPI app with WebSocket chat endpoint.
Serves a single-page chat UI and handles real-time messaging
through the same handle_message() pipeline as WhatsApp.
"""

import hashlib
import hmac
import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse

import config
from agent import handle_message

log = logging.getLogger(__name__)


def create_app(molly) -> FastAPI:
    """Create the FastAPI app wired to the Molly instance."""
    app = FastAPI(title="Molly", docs_url=None, redoc_url=None)

    html_path = Path(__file__).parent / "web" / "index.html"

    @app.get("/")
    async def index():
        """Serve the chat UI."""
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>Molly Web UI</h1><p>web/index.html not found</p>", status_code=500)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket, token: str = Query("")):
        # Token auth (constant-time comparison)
        if config.WEB_AUTH_TOKEN and not hmac.compare_digest(token, config.WEB_AUTH_TOKEN):
            await ws.accept()
            await ws.close(code=4003, reason="Invalid token")
            return

        await ws.accept()
        # Use stable session key based on token hash so reconnects preserve context.
        # Falls back to random ID when no token is configured.
        if config.WEB_AUTH_TOKEN:
            stable_id = hashlib.sha256(token.encode()).hexdigest()[:8]
        else:
            stable_id = uuid.uuid4().hex[:8]
        session_key = f"web:{stable_id}"
        log.info("Web client connected: %s", session_key)

        try:
            while True:
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    msg = {"text": data}

                text = msg.get("text", "").strip()
                if not text:
                    continue
                if len(text) > 4096:
                    await ws.send_json({"type": "message", "text": "Message too long (max 4096 chars)."})
                    continue

                # Send typing indicator
                await ws.send_json({"type": "typing"})

                # Get existing session for this connection
                session_id = molly.sessions.get(session_key)

                try:
                    response, new_session_id = await handle_message(
                        text, session_key, session_id,
                        approval_manager=molly.approvals,
                        molly_instance=molly,
                        source="web",
                    )

                    if new_session_id:
                        molly.sessions[session_key] = new_session_id
                        molly.save_sessions()

                    await ws.send_json({
                        "type": "message",
                        "text": response or "(no response)",
                    })
                except Exception:
                    log.error("Web message handling failed for %s", session_key, exc_info=True)
                    await ws.send_json({
                        "type": "message",
                        "text": "Something went wrong on my end. Try again in a moment.",
                    })

        except WebSocketDisconnect:
            log.info("Web client disconnected: %s", session_key)
        except Exception:
            log.error("WebSocket error for %s", session_key, exc_info=True)

    try:
        from gateway import attach_gateway_routes

        attach_gateway_routes(app, molly)
    except Exception:
        log.debug("Gateway routes unavailable", exc_info=True)

    return app
