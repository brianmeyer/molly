"""Google OAuth token management for Molly.

Handles:
- Loading client secrets from ~/.molly/credentials/client_secret.json
- Loading/refreshing tokens from ~/.molly/credentials/token.json
- First-time browser-based OAuth flow
- Auto-refresh using refresh token
"""

import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import config

log = logging.getLogger(__name__)

# Module-level cache
_credentials: Credentials | None = None
_calendar_service = None
_gmail_service = None


def get_credentials() -> Credentials:
    """Get valid Google OAuth credentials, refreshing or re-authorizing as needed."""
    global _credentials

    if _credentials and _credentials.valid:
        return _credentials

    # Try loading existing token
    if config.GOOGLE_TOKEN.exists():
        _credentials = Credentials.from_authorized_user_file(
            str(config.GOOGLE_TOKEN), config.GOOGLE_SCOPES
        )

    # Refresh if expired
    if _credentials and _credentials.expired and _credentials.refresh_token:
        try:
            _credentials.refresh(Request())
            _save_token(_credentials)
            log.info("Google token refreshed")
            return _credentials
        except Exception:
            log.warning("Token refresh failed, need re-authorization", exc_info=True)
            _credentials = None

    # No valid credentials — run OAuth flow
    if not _credentials or not _credentials.valid:
        if not config.GOOGLE_CLIENT_SECRET.exists():
            raise FileNotFoundError(
                f"Google client secret not found at {config.GOOGLE_CLIENT_SECRET}. "
                f"Download it from Google Cloud Console."
            )

        log.info("Starting Google OAuth flow (browser will open)...")
        flow = InstalledAppFlow.from_client_secrets_file(
            str(config.GOOGLE_CLIENT_SECRET), config.GOOGLE_SCOPES
        )
        _credentials = flow.run_local_server(port=0)
        _save_token(_credentials)
        log.info("Google OAuth completed and token saved")

    return _credentials


def _save_token(creds: Credentials):
    """Persist credentials to token.json."""
    config.GOOGLE_CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    config.GOOGLE_TOKEN.write_text(creds.to_json())


def get_calendar_service():
    """Get a cached Google Calendar API service instance."""
    global _calendar_service
    if _calendar_service is None:
        creds = get_credentials()
        _calendar_service = build("calendar", "v3", credentials=creds)
    return _calendar_service


def get_gmail_service():
    """Get a cached Gmail API service instance."""
    global _gmail_service
    if _gmail_service is None:
        creds = get_credentials()
        _gmail_service = build("gmail", "v1", credentials=creds)
    return _gmail_service


def reset_services():
    """Clear cached services (call after token refresh issues)."""
    global _credentials, _calendar_service, _gmail_service
    _credentials = None
    _calendar_service = None
    _gmail_service = None
