import os
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from main import app
from integrations.api import UPSTREAM_API_URL, API_KEY

client = TestClient(app)

# --- Mocked Unit Tests ---

@patch("httpx.AsyncClient.get")
def test_get_models_mocked_success(mock_get):
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "model-1", "object": "model"},
            {"id": "model-2", "object": "model"}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    
    # Assign the mock response to be returned by get
    mock_get.return_value = mock_response

    response = client.get("/api/models")
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 2
    assert data["data"][0]["id"] == "model-1"
    mock_get.assert_called_once()


@patch("httpx.AsyncClient.get")
def test_get_models_mocked_upstream_error(mock_get):
    # Mock response with error
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    
    # We raise HTTPStatusError when raise_for_status is called
    req = httpx.Request("GET", f"{UPSTREAM_API_URL}/models")
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="Unauthorized",
        request=req,
        response=mock_response
    )
    
    mock_get.return_value = mock_response

    response = client.get("/api/models")
    
    assert response.status_code == 401
    assert "Upstream error" in response.json()["detail"]


@patch("httpx.AsyncClient.post")
def test_chat_non_stream_mocked_success(mock_post):
    # Mock response for non-stream
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! I am an LLM."
                },
                "finish_reason": "stop"
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False
    }
    
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello! I am an LLM."
    mock_post.assert_called_once()


class AsyncContextManagerMock:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@patch("httpx.AsyncClient.stream")
def test_chat_stream_mocked_success(mock_stream):
    # Mock for async response
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    # Mock the async generator for response.aiter_lines
    async def mock_aiter_lines():
        lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            'data: [DONE]'
        ]
        for line in lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines
    mock_stream.return_value = AsyncContextManagerMock(mock_response)

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True
    }
    
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 200
    content = response.text
    assert "Hello" in content
    assert "world" in content
    assert "[DONE]" in content


@patch("httpx.AsyncClient.stream")
def test_chat_stream_mocked_upstream_error(mock_stream):
    # Mock for response that returns an error status code
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    # Mock the aread method to return bytes error
    mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
    mock_stream.return_value = AsyncContextManagerMock(mock_response)

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True
    }
    
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 200 # StreamingResponse in api.py catches exceptions inside generator and yields error messages in event stream, keeping response status 200
    assert "Upstream error 500" in response.text


# --- Live Integration Tests (Direct LLM API Verification) ---

# We only run these if API_KEY is set (meaning we have credentials to make real requests)
API_KEY_PRESENT = bool(API_KEY and API_KEY != "sk-pass55")

@pytest.mark.skipif(not API_KEY_PRESENT, reason="API_KEY is not configured with a real key")
def test_live_upstream_api_connection():
    """
    Directly tests the connection to the upstream LLM API.
    This verifies if the upstream API is working and if the credentials are valid.
    """
    print(f"\nTesting upstream connection directly to: {UPSTREAM_API_URL}")
    with httpx.Client() as sync_client:
        try:
            response = sync_client.get(
                f"{UPSTREAM_API_URL}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=15.0
            )
            print(f"Direct connection status code: {response.status_code}")
            assert response.status_code == 200, f"Failed to connect to upstream. Status: {response.status_code}, Response: {response.text}"
            models_data = response.json()
            assert "data" in models_data, "Invalid models response format from upstream"
            print("Successfully retrieved models from upstream:")
            models = [m.get("id") for m in models_data.get("data", [])]
            print(f"Available models: {models}")
        except Exception as e:
            pytest.fail(f"Direct connection failed: {e}")


@pytest.mark.skipif(not API_KEY_PRESENT, reason="API_KEY is not configured with a real key")
def test_live_proxy_models_endpoint():
    """
    Tests our FastAPI proxy models endpoint to see if it correctly forwards the response from upstream.
    """
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0


@pytest.mark.skipif(not API_KEY_PRESENT, reason="API_KEY is not configured with a real key")
def test_live_proxy_chat_completion():
    """
    Tests our FastAPI proxy chat completion endpoint with stream=False.
    """
    # First, list models to find a valid one
    models_response = client.get("/api/models")
    assert models_response.status_code == 200
    models_data = models_response.json()
    assert "data" in models_data and len(models_data["data"]) > 0
    
    # Use the first available model
    model_name = models_data["data"][0]["id"]
    print(f"\nTesting live chat completion with model: {model_name}")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Respond with the single word 'Success'."}
        ],
        "stream": False,
        "temperature": 0.0
    }
    
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    print(f"Response from chat completion: {content.strip()}")
    assert len(content.strip()) > 0
