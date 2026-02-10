"""FastAPI server for WhisperX OpenAI Bridge"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
import tempfile
import os
from pathlib import Path
import logging

from .config import config
from .transcription import WhisperXService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WhisperX OpenAI Bridge",
    description="OpenAI-compatible API for WhisperX speech recognition",
    version="0.1.0"
)

# Initialize WhisperX service
whisperx_service = WhisperXService(
    model_name=config.whisperx.model,
    device=config.whisperx.device,
    compute_type=config.whisperx.compute_type,
    batch_size=config.whisperx.batch_size,
    hf_token=config.whisperx.hf_token
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Loading WhisperX model...")
    whisperx_service.load_model()
    logger.info("WhisperX model loaded successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": config.whisperx.model}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "WhisperX OpenAI Bridge",
        "version": "0.1.0",
        "endpoints": {
            "transcribe": "/v1/audio/transcriptions",
            "health": "/health"
        }
    }


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    timestamp_granularities: Optional[str] = Form(None)
):
    """
    Create transcription from audio file (OpenAI-compatible endpoint)
    
    Args:
        file: Audio file to transcribe
        model: Model to use (required by OpenAI API spec, but ignored - uses configured model)
        language: Language of the audio (optional)
        prompt: Initial prompt to guide the transcription (optional)
        response_format: Format of the response (json, text, srt, vtt, verbose_json)
        temperature: Sampling temperature (optional, currently not used)
        timestamp_granularities: Comma-separated list of timestamp types (word, segment)
    
    Returns:
        Transcription result in specified format
    """
    logger.info(f"Received transcription request: file={file.filename}, language={language}")
    
    # Parse timestamp granularities
    granularities = None
    if timestamp_granularities:
        granularities = [g.strip() for g in timestamp_granularities.split(",")]
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename or "audio")
    
    try:
        # Write uploaded file to temp location
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        logger.info(f"Saved audio to temporary file: {temp_file_path}")
        
        # Transcribe
        result = whisperx_service.transcribe(
            audio_path=temp_file_path,
            language=language,
            initial_prompt=prompt,
            word_timestamps=(granularities and "word" in granularities) or False
        )
        
        logger.info(f"Transcription completed successfully")
        
        # Format response
        formatted_result = whisperx_service.format_response(
            result=result,
            response_format=response_format,
            timestamp_granularities=granularities
        )
        
        return JSONResponse(content=formatted_result)
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {str(e)}")


@app.post("/v1/audio/transcriptions/diarize")
async def create_transcription_with_diarization(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None)
):
    """
    Create transcription with speaker diarization
    
    Args:
        file: Audio file to transcribe
        model: Model to use (ignored)
        language: Language of the audio (optional)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
    
    Returns:
        Transcription result with speaker labels
    """
    logger.info(f"Received diarization request: file={file.filename}")
    
    # Check if HF token is configured
    if not config.whisperx.hf_token:
        raise HTTPException(
            status_code=400,
            detail="Diarization requires HuggingFace token. Please configure HF_TOKEN."
        )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename or "audio")
    
    try:
        # Write uploaded file to temp location
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Transcribe with diarization
        result = whisperx_service.transcribe_with_diarization(
            audio_path=temp_file_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        logger.info(f"Diarization completed successfully")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Diarization error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")
        
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {str(e)}")


def run_server():
    """Run the server"""
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
