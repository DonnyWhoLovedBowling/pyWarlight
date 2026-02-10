"""Configuration management for WhisperX OpenAI Bridge"""

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class WhisperXConfig(BaseModel):
    """WhisperX model configuration"""
    model: str = os.getenv("WHISPERX_MODEL", "base")
    device: str = os.getenv("WHISPERX_DEVICE", "cpu")
    compute_type: str = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")
    batch_size: int = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
    hf_token: Optional[str] = os.getenv("HF_TOKEN")


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))


class Config(BaseModel):
    """Main application configuration"""
    whisperx: WhisperXConfig = WhisperXConfig()
    server: ServerConfig = ServerConfig()


# Global config instance
config = Config()
