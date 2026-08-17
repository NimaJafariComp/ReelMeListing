"""ASGI entry point; bind Uvicorn to 127.0.0.1 for the local-only service."""

from listing_to_reel.api.service import create_app

app = create_app()
