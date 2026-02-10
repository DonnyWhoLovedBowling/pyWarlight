"""WhisperX transcription service"""

import whisperx
import torch
from typing import Optional, Dict, Any, List
import tempfile
import os
from pathlib import Path


class WhisperXService:
    """Service for handling WhisperX transcription"""
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 16,
        hf_token: Optional[str] = None
    ):
        """
        Initialize WhisperX service
        
        Args:
            model_name: WhisperX model name (tiny, base, small, medium, large-v2, large-v3)
            device: Device to run on (cpu, cuda)
            compute_type: Compute type (int8, float16, float32)
            batch_size: Batch size for processing
            hf_token: HuggingFace token for diarization
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.model = None
        self.align_model = None
        self.align_metadata = None
        
    def load_model(self):
        """Load WhisperX model"""
        if self.model is None:
            self.model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type
            )
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es')
            initial_prompt: Initial prompt for the model
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Dictionary with transcription results
        """
        # Load model if not already loaded
        self.load_model()
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe with WhisperX
        result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=language,
            initial_prompt=initial_prompt
        )
        
        # Align whisper output if word timestamps requested
        if word_timestamps and result.get("segments"):
            detected_language = result.get("language", language or "en")
            
            # Load alignment model
            if self.align_model is None or self.align_metadata is None:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.device
                )
            
            # Align
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
        
        return result
    
    def transcribe_with_diarization(
        self,
        audio_path: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Transcribe with speaker diarization
        
        Args:
            audio_path: Path to audio file
            language: Language code
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            Dictionary with transcription and diarization results
        """
        if not self.hf_token:
            raise ValueError("HuggingFace token required for diarization")
        
        # Get base transcription
        result = self.transcribe(audio_path, language=language, word_timestamps=True)
        
        # Load audio for diarization
        audio = whisperx.load_audio(audio_path)
        
        # Diarize
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token,
            device=self.device
        )
        
        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Assign speaker labels
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        return result
    
    def format_response(
        self,
        result: Dict[str, Any],
        response_format: str = "json",
        timestamp_granularities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Format transcription result according to OpenAI format
        
        Args:
            result: WhisperX result
            response_format: Response format (json, text, srt, vtt, verbose_json)
            timestamp_granularities: List of timestamp types to include
            
        Returns:
            Formatted response
        """
        text = " ".join([segment.get("text", "") for segment in result.get("segments", [])])
        
        if response_format == "text":
            return {"text": text}
        
        response = {
            "text": text,
            "language": result.get("language", "en")
        }
        
        # Add segments if requested
        if timestamp_granularities and "segment" in timestamp_granularities:
            response["segments"] = result.get("segments", [])
        
        # Add word-level timestamps if requested
        if timestamp_granularities and "word" in timestamp_granularities:
            words = []
            for segment in result.get("segments", []):
                if "words" in segment:
                    words.extend(segment["words"])
            if words:
                response["words"] = words
        
        return response
