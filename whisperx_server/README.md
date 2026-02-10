# WhisperX OpenAI-Compatible Transcription Server

This server provides an OpenAI-compatible API endpoint for audio transcription using WhisperX, which adds speaker diarization (voice identification) capabilities to the Whisper model.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint
- **Speaker Diarization**: Identifies different speakers in the audio (requires HuggingFace token)
- **Multiple Output Formats**: Supports JSON, text, SRT, VTT, and verbose JSON formats
- **Model Caching**: Efficient model loading and reuse across requests
- **GPU Support**: Automatically uses GPU if available for faster transcription

## Installation

1. Install the required dependencies:

```bash
pip install -r whisperx_server/requirements.txt
```

2. (Optional) For speaker diarization, you need a HuggingFace token:
   - Create an account at https://huggingface.co/
   - Accept the terms for pyannote models at https://huggingface.co/pyannote/speaker-diarization
   - Get your token from https://huggingface.co/settings/tokens
   - Set the `HF_TOKEN` environment variable

## Usage

### Starting the Server

```bash
# Basic usage
python -m whisperx_server.server

# Or with uvicorn directly
uvicorn whisperx_server.server:app --host 0.0.0.0 --port 8000

# With custom settings
export WHISPER_MODEL=medium
export HF_TOKEN=your_huggingface_token
export PORT=8000
python -m whisperx_server.server
```

### Environment Variables

- `WHISPER_MODEL`: WhisperX model to use (default: "base")
  - Options: tiny, base, small, medium, large, large-v2, large-v3
- `HF_TOKEN`: HuggingFace token for speaker diarization (optional)
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8000)
- `BATCH_SIZE`: Batch size for transcription (default: 16)

### Making Requests

The server is compatible with OpenAI's transcription API. Here are some examples:

#### Using curl

```bash
# Basic transcription
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"

# With language specification
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=en"

# Get SRT subtitles
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=srt"

# Get verbose JSON with segments and speaker labels
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json"
```

#### Using Python

```python
import requests

# Transcribe audio file
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        files={"file": f},
        data={"model": "whisper-1", "response_format": "json"}
    )
    
print(response.json())
```

#### Using OpenAI Python Client

```python
from openai import OpenAI

# Point the OpenAI client to your local server
client = OpenAI(
    api_key="dummy-key",  # Not used but required by the client
    base_url="http://localhost:8000/v1"
)

# Use the client as normal
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    
print(transcript.text)
```

## Response Formats

### JSON (default)
```json
{
  "text": "This is the transcribed text."
}
```

### Verbose JSON
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 10.5,
  "text": "This is the transcribed text.",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "This is the transcribed text.",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

### Text
Plain text output of the transcription.

### SRT
Standard SubRip subtitle format.

### VTT
WebVTT subtitle format with speaker labels (if diarization is enabled).

## Integration with OpenWebUI

To integrate this server with OpenWebUI:

1. Start the WhisperX server:
```bash
python -m whisperx_server.server
```

2. In OpenWebUI settings, configure the transcription service:
   - Set the API endpoint to: `http://localhost:8000/v1`
   - Set the API key to any dummy value (it's not validated)
   - The server will now be used for all transcription requests

## API Endpoints

- `GET /`: Server information
- `GET /health`: Health check endpoint
- `POST /v1/audio/transcriptions`: Transcribe audio (OpenAI compatible)

## Supported Audio Formats

The server supports all audio formats that WhisperX/ffmpeg can handle, including:
- MP3
- WAV
- M4A
- FLAC
- OGG
- And many more

## Performance Tips

1. **Use GPU**: The server automatically detects and uses CUDA if available
2. **Choose the right model**: Larger models are more accurate but slower
   - `tiny`: Fastest, least accurate
   - `base`: Good balance for most use cases
   - `small`: Better accuracy, still reasonably fast
   - `medium`: High accuracy, slower
   - `large-v3`: Best accuracy, slowest
3. **Adjust batch size**: Increase `BATCH_SIZE` if you have more GPU memory

## Troubleshooting

### "HF_TOKEN not set" warning
This is normal if you don't need speaker diarization. The server will work fine without it, just without speaker labels.

### CUDA out of memory
Try a smaller model or reduce the batch size:
```bash
export WHISPER_MODEL=base
export BATCH_SIZE=8
```

### Slow transcription
- Use a GPU if available
- Use a smaller model
- Increase batch size (if you have enough memory)

## License

This server is provided as-is. WhisperX and its dependencies have their own licenses.
