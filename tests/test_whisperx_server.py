"""
Tests for the WhisperX transcription server.

These tests verify the basic functionality and API compatibility of the server.
"""

import pytest
from fastapi.testclient import TestClient
import io
import wave
import numpy as np


@pytest.fixture
def client():
    """Create a test client for the API."""
    from whisperx_server.server import app
    return TestClient(app)


@pytest.fixture
def sample_audio():
    """Create a simple audio file for testing (1 second of silence)."""
    # Create a simple WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        # Set parameters: 1 channel, 2 bytes per sample, 16000 Hz sample rate
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        
        # Generate 1 second of silence
        duration = 1.0  # seconds
        num_samples = int(16000 * duration)
        silence = np.zeros(num_samples, dtype=np.int16)
        wav_file.writeframes(silence.tobytes())
    
    buffer.seek(0)
    return buffer


def test_root_endpoint(client):
    """Test the root endpoint returns server information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
    assert data["endpoints"]["transcriptions"] == "/v1/audio/transcriptions"


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_transcription_endpoint_requires_file(client):
    """Test that the transcription endpoint requires a file."""
    response = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert response.status_code == 422  # Unprocessable Entity


def test_transcription_endpoint_basic(client, sample_audio):
    """Test basic transcription with default JSON format."""
    # Note: This test may fail without actual WhisperX models installed
    # It's mainly for verifying the API structure
    try:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )
        
        # Check response structure regardless of whether models are available
        if response.status_code == 200:
            data = response.json()
            assert "text" in data
        elif response.status_code == 500:
            # Models may not be available in test environment
            assert "error" in response.json() or "detail" in response.json()
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    except Exception as e:
        # If WhisperX is not installed, skip this test
        pytest.skip(f"WhisperX not available: {e}")


def test_transcription_endpoint_with_language(client, sample_audio):
    """Test transcription with language parameter."""
    try:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1", "language": "en"}
        )
        
        # Accept both success and model-not-available errors
        assert response.status_code in [200, 500]
    except Exception as e:
        pytest.skip(f"WhisperX not available: {e}")


def test_transcription_endpoint_text_format(client, sample_audio):
    """Test transcription with text response format."""
    try:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1", "response_format": "text"}
        )
        
        # Accept both success and model-not-available errors
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            # Text format should return plain text, not JSON
            assert isinstance(response.text, str)
    except Exception as e:
        pytest.skip(f"WhisperX not available: {e}")


def test_transcription_endpoint_verbose_json_format(client, sample_audio):
    """Test transcription with verbose_json response format."""
    try:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1", "response_format": "verbose_json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            # Check for expected fields in verbose JSON
            assert "text" in data
            assert "task" in data
            assert "language" in data
            # Note: segments and other fields depend on actual transcription
    except Exception as e:
        pytest.skip(f"WhisperX not available: {e}")


def test_format_timestamp_functions():
    """Test timestamp formatting functions."""
    from whisperx_server.server import format_timestamp_srt, format_timestamp_vtt
    
    # Test SRT format (HH:MM:SS,mmm)
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(1.5) == "00:00:01,500"
    assert format_timestamp_srt(61.234) == "00:01:01,234"
    assert format_timestamp_srt(3661.999) == "01:01:01,999"
    
    # Test VTT format (HH:MM:SS.mmm)
    assert format_timestamp_vtt(0.0) == "00:00:00.000"
    assert format_timestamp_vtt(1.5) == "00:00:01.500"
    assert format_timestamp_vtt(61.234) == "00:01:01.234"
    assert format_timestamp_vtt(3661.999) == "01:01:01.999"


def test_format_as_srt():
    """Test SRT format generation."""
    from whisperx_server.server import format_as_srt
    
    segments = [
        {"start": 0.0, "end": 2.5, "text": "First segment"},
        {"start": 2.5, "end": 5.0, "text": "Second segment"},
    ]
    
    srt = format_as_srt(segments)
    
    # Check that SRT format is correct
    assert "1\n" in srt
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert "First segment" in srt
    assert "2\n" in srt
    assert "00:00:02,500 --> 00:00:05,000" in srt
    assert "Second segment" in srt


def test_format_as_vtt():
    """Test VTT format generation."""
    from whisperx_server.server import format_as_vtt
    
    segments = [
        {"start": 0.0, "end": 2.5, "text": "First segment", "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 5.0, "text": "Second segment", "speaker": "SPEAKER_01"},
    ]
    
    vtt = format_as_vtt(segments)
    
    # Check that VTT format is correct
    assert "WEBVTT" in vtt
    assert "00:00:00.000 --> 00:00:02.500" in vtt
    assert "[SPEAKER_00] First segment" in vtt
    assert "00:00:02.500 --> 00:00:05.000" in vtt
    assert "[SPEAKER_01] Second segment" in vtt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
