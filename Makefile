.PHONY: run run-mps web-build test lint

# Build the bundled React studio, then serve it and the local API together.
run: web-build
	uv run uvicorn listing_to_reel.api.app:app --host 127.0.0.1 --port 8000

# Apple Silicon image editing requires the MPS extras.
run-mps: web-build
	uv run --extra mps uvicorn listing_to_reel.api.app:app --host 127.0.0.1 --port 8000

web-build:
	npm --prefix webapp run build

lint:
	uv run --extra dev ruff check .

test:
	uv run --extra dev pytest
