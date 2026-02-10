"""Tests for server endpoints"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from src.whisperx_bridge.server import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data


@patch("src.whisperx_bridge.server.whisperx_service")
def test_transcription_endpoint_missing_file(mock_service, client):
    """Test transcription endpoint with missing file"""
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1"}
    )
    assert response.status_code == 422  # Validation error


@patch("src.whisperx_bridge.server.whisperx_service")
@patch("builtins.open", create=True)
@patch("os.path.exists")
@patch("os.remove")
@patch("os.rmdir")
def test_transcription_endpoint_success(mock_rmdir, mock_remove, mock_exists, mock_open, mock_service, client):
    """Test successful transcription"""
    # Mock the transcription service
    mock_service.transcribe.return_value = {
        "segments": [{"text": "Hello world"}],
        "language": "en"
    }
    mock_service.format_response.return_value = {
        "text": "Hello world",
        "language": "en"
    }
    
    # Mock file operations
    mock_exists.return_value = True
    mock_open.return_value.__enter__.return_value.write = Mock()
    
    # Create a mock file
    audio_content = b"fake audio content"
    files = {"file": ("test.wav", audio_content, "audio/wav")}
    data = {"model": "whisper-1", "language": "en"}
    
    response = client.post(
        "/v1/audio/transcriptions",
        files=files,
        data=data
    )
    
    # With proper mocking, this should succeed
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert result["text"] == "Hello world"
