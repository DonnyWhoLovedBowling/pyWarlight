"""Tests for configuration module"""

import os
import pytest
from src.whisperx_bridge.config import Config, WhisperXConfig, ServerConfig


def test_whisperx_config_defaults():
    """Test WhisperX config with default values"""
    config = WhisperXConfig()
    assert config.model in ["base", "tiny", "small", "medium", "large-v2", "large-v3"]
    assert config.device in ["cpu", "cuda"]
    assert config.compute_type in ["int8", "float16", "float32"]
    assert config.batch_size > 0


def test_server_config_defaults():
    """Test server config with default values"""
    config = ServerConfig()
    assert isinstance(config.host, str)
    assert isinstance(config.port, int)
    assert config.port > 0
    assert config.port < 65536


def test_main_config():
    """Test main config"""
    config = Config()
    assert isinstance(config.whisperx, WhisperXConfig)
    assert isinstance(config.server, ServerConfig)


def test_config_with_env_vars(monkeypatch):
    """Test config with environment variables"""
    monkeypatch.setenv("WHISPERX_MODEL", "small")
    monkeypatch.setenv("WHISPERX_DEVICE", "cuda")
    monkeypatch.setenv("SERVER_PORT", "9000")
    
    config = WhisperXConfig()
    assert config.model == "small"
    assert config.device == "cuda"
    
    server_config = ServerConfig()
    assert server_config.port == 9000
