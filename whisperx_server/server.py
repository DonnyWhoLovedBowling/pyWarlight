"""
OpenAI-compatible API server for WhisperX transcription.

This server provides an OpenAI-compatible endpoint for audio transcription
using WhisperX, which adds speaker diarization capabilities to the Whisper model.
"""

import os
import tempfile
import logging
from typing import Optional, Literal
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import whisperx
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhisperX Transcription API",
    description="OpenAI-compatible API for audio transcription using WhisperX",
    version="0.1.0",
)

# Global variables for model caching
_model = None
_diarize_model = None
_model_name = None
_device = None
_compute_type = None


def initialize_models(
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8"
):
    """Initialize WhisperX models."""
    global _model, _diarize_model, _model_name, _device, _compute_type
    
    if _model is None or _model_name != model_name:
        logger.info(f"Loading WhisperX model: {model_name} on {device}")
        _model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type
        )
        _model_name = model_name
        _device = device
        _compute_type = compute_type
        logger.info("Model loaded successfully")
    
    return _model


def get_diarize_model(device: str = "cpu"):
    """Get or initialize diarization model."""
    global _diarize_model
    
    if _diarize_model is None:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("Loading diarization model")
            _diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )
            logger.info("Diarization model loaded successfully")
        else:
            logger.warning("HF_TOKEN not set. Diarization will be disabled.")
    
    return _diarize_model


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "WhisperX Transcription API",
        "version": "0.1.0",
        "endpoints": {
            "transcriptions": "/v1/audio/transcriptions"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: Optional[str] = Form(default=None),
):
    """
    Transcribe audio file using WhisperX.
    
    This endpoint is compatible with OpenAI's audio transcription API.
    
    Args:
        file: The audio file to transcribe
        model: Model to use (ignored, using WhisperX)
        language: Language code (e.g., 'en', 'es')
        prompt: Optional text to guide the model
        response_format: Format of the response
        temperature: Sampling temperature (0-1)
        timestamp_granularities: Timestamp granularities (not used in this implementation)
    
    Returns:
        Transcription response in the requested format
    """
    logger.info(f"Received transcription request for file: {file.filename}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Determine device and compute type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    # Get model name from environment or use default
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    
    # Initialize model
    model_instance = initialize_models(whisper_model, device, compute_type)
    
    # Save uploaded file to temporary location
    temp_file = None
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing audio file: {temp_file_path}")
        
        # Load audio
        audio = whisperx.load_audio(temp_file_path)
        
        # Transcribe with WhisperX
        result = model_instance.transcribe(
            audio,
            batch_size=int(os.getenv("BATCH_SIZE", "16")),
            language=language
        )
        
        # Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get("language", language or "en"),
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
        
        # Perform diarization if enabled
        diarize_model = get_diarize_model(device)
        if diarize_model:
            try:
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info("Diarization completed")
            except Exception as e:
                logger.warning(f"Diarization failed: {e}. Continuing without speaker labels.")
        
        # Format response based on requested format
        if response_format == "text":
            # Simple text format
            text = " ".join([segment.get("text", "") for segment in result.get("segments", [])])
            return text
        
        elif response_format == "srt":
            # SRT subtitle format
            srt_content = format_as_srt(result.get("segments", []))
            return srt_content
        
        elif response_format == "vtt":
            # WebVTT subtitle format
            vtt_content = format_as_vtt(result.get("segments", []))
            return vtt_content
        
        elif response_format == "verbose_json":
            # Verbose JSON format with all details
            return JSONResponse(content={
                "task": "transcribe",
                "language": result.get("language", language or "en"),
                "duration": result.get("segments", [{}])[-1].get("end", 0.0) if result.get("segments") else 0.0,
                "text": " ".join([segment.get("text", "") for segment in result.get("segments", [])]),
                "segments": result.get("segments", []),
                "words": result.get("word_segments", []),
            })
        
        else:  # json (default)
            # Standard JSON format (OpenAI compatible)
            text = " ".join([segment.get("text", "") for segment in result.get("segments", [])])
            return JSONResponse(content={
                "text": text
            })
    
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info("Temporary file cleaned up")


def format_as_srt(segments: list) -> str:
    """Format segments as SRT subtitle format."""
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp_srt(segment.get("start", 0.0))
        end = format_timestamp_srt(segment.get("end", 0.0))
        text = segment.get("text", "").strip()
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def format_as_vtt(segments: list) -> str:
    """Format segments as WebVTT subtitle format."""
    vtt_lines = ["WEBVTT", ""]
    
    for segment in segments:
        start = format_timestamp_vtt(segment.get("start", 0.0))
        end = format_timestamp_vtt(segment.get("end", 0.0))
        text = segment.get("text", "").strip()
        
        # Add speaker label if available
        speaker = segment.get("speaker")
        if speaker:
            text = f"[{speaker}] {text}"
        
        vtt_lines.append(f"{start} --> {end}")
        vtt_lines.append(text)
        vtt_lines.append("")
    
    return "\n".join(vtt_lines)


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting WhisperX Transcription API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
