# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- ffmpeg (for audio processing)
- CUDA (optional, for GPU acceleration)

## Installation

1. **Clone and navigate to the project:**
   ```bash
   cd WhisperXOpenAIBridge
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the server:**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

## Running the Server

### Basic Usage

Start the server with default settings:
```bash
python -m src.whisperx_bridge.server
```

The server will start on `http://localhost:8000`

### Using Docker

Build and run with Docker Compose:
```bash
docker-compose up --build
```

### Using Docker (manual)

Build the image:
```bash
docker build -t whisperx-bridge .
```

Run the container:
```bash
docker run -p 8000:8000 \
  -e WHISPERX_MODEL=base \
  -e WHISPERX_DEVICE=cpu \
  whisperx-bridge
```

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Transcribe an audio file
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@your-audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"
```

### Using the example script

```bash
python example_usage.py path/to/audio.mp3
```

### Using Python requests

```python
import requests

url = "http://localhost:8000/v1/audio/transcriptions"
files = {"file": open("audio.mp3", "rb")}
data = {"model": "whisper-1", "language": "en"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPERX_MODEL` | `base` | Model size (tiny, base, small, medium, large-v2, large-v3) |
| `WHISPERX_DEVICE` | `cpu` | Device to use (cpu, cuda) |
| `WHISPERX_COMPUTE_TYPE` | `int8` | Compute type (int8, float16, float32) |
| `WHISPERX_BATCH_SIZE` | `16` | Batch size for processing |
| `SERVER_HOST` | `0.0.0.0` | Server host address |
| `SERVER_PORT` | `8000` | Server port |
| `HF_TOKEN` | - | HuggingFace token (required for diarization) |

### Model Sizes

- **tiny**: Fastest, least accurate (~1GB RAM)
- **base**: Good balance (~2GB RAM)
- **small**: Better accuracy (~5GB RAM)
- **medium**: High accuracy (~10GB RAM)
- **large-v2/v3**: Best accuracy (~10GB RAM, slower)

## API Endpoints

### POST /v1/audio/transcriptions

OpenAI-compatible transcription endpoint.

**Parameters:**
- `file` (required): Audio file
- `model` (required): Model identifier (e.g., "whisper-1")
- `language` (optional): Language code (e.g., "en", "es")
- `prompt` (optional): Initial prompt
- `response_format` (optional): json, text, srt, vtt, verbose_json
- `timestamp_granularities` (optional): "word", "segment", or "word,segment"

### POST /v1/audio/transcriptions/diarize

Transcription with speaker diarization (requires HF_TOKEN).

**Parameters:**
- `file` (required): Audio file
- `model` (required): Model identifier
- `language` (optional): Language code
- `min_speakers` (optional): Minimum number of speakers
- `max_speakers` (optional): Maximum number of speakers

### GET /health

Health check endpoint.

### GET /

Root endpoint with API information.

## Troubleshooting

### Common Issues

1. **Module not found errors**
   - Make sure you've activated your virtual environment
   - Run `pip install -r requirements.txt`

2. **CUDA out of memory**
   - Use a smaller model (e.g., "base" instead of "large")
   - Set `WHISPERX_DEVICE=cpu`
   - Reduce `WHISPERX_BATCH_SIZE`

3. **ffmpeg not found**
   - Install ffmpeg: `sudo apt-get install ffmpeg` (Ubuntu/Debian)
   - Or download from https://ffmpeg.org/

4. **Diarization not working**
   - Set `HF_TOKEN` in your `.env` file
   - Get a token from https://huggingface.co/settings/tokens

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=src/whisperx_bridge
```

### Code Style

```bash
# Format code
black src/ tests/

# Verify structure
python verify_structure.py
```

## Performance Tips

1. **Use GPU**: Set `WHISPERX_DEVICE=cuda` for 10-20x speedup
2. **Choose right model**: Balance accuracy vs speed
3. **Batch processing**: Use higher `WHISPERX_BATCH_SIZE` for batch jobs
4. **Compute type**: Use `float16` on GPU for faster inference

## Support

For issues and questions:
- Check the README.md
- Review the example_usage.py
- Check WhisperX documentation: https://github.com/m-bain/whisperX
