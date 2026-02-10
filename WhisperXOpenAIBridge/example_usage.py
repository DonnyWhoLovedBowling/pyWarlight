"""Example usage of WhisperX OpenAI Bridge"""

import requests
import json

# Server URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:", response.json())


def transcribe_audio(audio_file_path: str, language: str = None):
    """
    Transcribe an audio file
    
    Args:
        audio_file_path: Path to audio file
        language: Optional language code (e.g., 'en', 'es')
    """
    url = f"{BASE_URL}/v1/audio/transcriptions"
    
    with open(audio_file_path, "rb") as audio_file:
        files = {"file": audio_file}
        data = {
            "model": "whisper-1",  # Required by OpenAI API spec
            "response_format": "json",
        }
        
        if language:
            data["language"] = language
        
        response = requests.post(url, files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        print("\n=== Transcription Result ===")
        print(f"Text: {result.get('text')}")
        print(f"Language: {result.get('language')}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def transcribe_with_timestamps(audio_file_path: str):
    """
    Transcribe with word and segment timestamps
    
    Args:
        audio_file_path: Path to audio file
    """
    url = f"{BASE_URL}/v1/audio/transcriptions"
    
    with open(audio_file_path, "rb") as audio_file:
        files = {"file": audio_file}
        data = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities": "word,segment"
        }
        
        response = requests.post(url, files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        print("\n=== Transcription with Timestamps ===")
        print(f"Text: {result.get('text')}")
        
        if "segments" in result:
            print("\nSegments:")
            for segment in result["segments"][:3]:  # Show first 3 segments
                print(f"  [{segment.get('start'):.2f}s - {segment.get('end'):.2f}s] {segment.get('text')}")
        
        if "words" in result:
            print("\nWords:")
            for word in result["words"][:10]:  # Show first 10 words
                print(f"  [{word.get('start'):.2f}s] {word.get('word')}")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


if __name__ == "__main__":
    import sys
    
    # Test health endpoint
    test_health()
    
    # Example usage
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"\nTranscribing: {audio_file}")
        
        # Basic transcription
        transcribe_audio(audio_file)
        
        # Transcription with timestamps
        transcribe_with_timestamps(audio_file)
    else:
        print("\nUsage: python example_usage.py <audio_file>")
        print("Example: python example_usage.py sample.wav")
