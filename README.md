# Noesis LLM Agent Client

A chat interface powered by FastAPI that proxies requests to an upstream LLM API.

## Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

To install dependencies:

```bash
uv sync
```

## Running the Application

To run the FastAPI server locally:

```bash
uv run main.py
```

The server will start at `http://localhost:8000`.

## Running Tests

We use `pytest` for unit and integration testing.

To run the entire test suite:

```bash
uv run pytest -v -s
```

### Test Suite Structure

- **Mocked Unit Tests**: Test local endpoints (`/api/models`, `/api/chat`) by mocking upstream HTTPX calls. These run instantly and do not require external network connections.
- **Live Integration Tests**: Make live calls directly to the configured `UPSTREAM_API_URL` to verify connectivity and API credentials. These are skipped automatically if no `API_KEY` is set in your `.env`.
