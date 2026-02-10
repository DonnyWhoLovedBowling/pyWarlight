"""
Example script to run the WhisperX transcription server.
"""

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print("=" * 60)
    print("WhisperX OpenAI-Compatible Transcription Server")
    print("=" * 60)
    print(f"Starting server on http://{host}:{port}")
    print(f"API endpoint: http://{host}:{port}/v1/audio/transcriptions")
    print(f"Model: {os.getenv('WHISPER_MODEL', 'base')}")
    print(f"Diarization: {'Enabled' if os.getenv('HF_TOKEN') else 'Disabled (set HF_TOKEN to enable)'}")
    print("=" * 60)
    
    uvicorn.run(
        "whisperx_server.server:app",
        host=host,
        port=port,
        log_level="info"
    )
