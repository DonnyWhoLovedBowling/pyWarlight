# WhisperXOpenAIBridge

A FastAPI server that bridges WhisperX speech recognition with OpenAI-compatible API endpoints.

## Overview

WhisperXOpenAIBridge provides an OpenAI-compatible API interface for WhisperX, allowing you to use WhisperX for speech recognition while maintaining compatibility with OpenAI's API format.

## Features

- OpenAI-compatible API endpoints for audio transcription
- Support for multiple audio formats
- Fast transcription using WhisperX
- Diarization support
- Word-level timestamps
- RESTful API with FastAPI

## Installation

```bash
cd WhisperXOpenAIBridge
pip install -r requirements.txt
```

## Usage

### Start the Server

```bash
python -m src.whisperx_bridge.server
```

The server will start on `http://localhost:8000` by default.

### API Endpoints

#### POST /v1/audio/transcriptions

Transcribe audio file (OpenAI-compatible endpoint)

**Request:**
- `file`: Audio file (multipart/form-data)
- `model`: Model name (e.g., "whisper-1")
- `language` (optional): Language code
- `response_format` (optional): Response format (json, text, srt, vtt)
- `timestamp_granularities` (optional): ["word", "segment"]

**Response:**
```json
{
  "text": "Transcribed text",
  "segments": [...],
  "language": "en"
}
```

#### GET /health

Health check endpoint

## Configuration

Configuration can be set via environment variables:

- `WHISPERX_MODEL`: WhisperX model to use (default: "base")
- `WHISPERX_DEVICE`: Device to use (default: "cpu")
- `WHISPERX_COMPUTE_TYPE`: Compute type (default: "int8")
- `SERVER_HOST`: Server host (default: "0.0.0.0")
- `SERVER_PORT`: Server port (default: 8000)

## Requirements

- Python 3.8+
- WhisperX
- FastAPI
- ffmpeg (for audio processing)

## License

MIT License
