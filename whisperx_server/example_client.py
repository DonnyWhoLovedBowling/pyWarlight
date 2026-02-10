"""
Example client script to test the WhisperX transcription server.

This demonstrates how to use the server with various methods.
"""

import requests
import sys
from pathlib import Path


def test_with_requests(audio_file: str, base_url: str = "http://localhost:8000"):
    """Test transcription using the requests library."""
    print(f"\n{'='*60}")
    print("Testing with requests library")
    print(f"{'='*60}")
    
    endpoint = f"{base_url}/v1/audio/transcriptions"
    
    # Test 1: Basic JSON response
    print("\n1. Basic JSON response:")
    with open(audio_file, "rb") as f:
        response = requests.post(
            endpoint,
            files={"file": f},
            data={"model": "whisper-1"}
        )
    
    if response.status_code == 200:
        print(f"✓ Success: {response.json()}")
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")
    
    # Test 2: Verbose JSON with segments
    print("\n2. Verbose JSON response:")
    with open(audio_file, "rb") as f:
        response = requests.post(
            endpoint,
            files={"file": f},
            data={"model": "whisper-1", "response_format": "verbose_json"}
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success:")
        print(f"  Language: {result.get('language')}")
        print(f"  Duration: {result.get('duration')} seconds")
        print(f"  Text: {result.get('text')}")
        print(f"  Segments: {len(result.get('segments', []))} segments")
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")
    
    # Test 3: Text format
    print("\n3. Plain text response:")
    with open(audio_file, "rb") as f:
        response = requests.post(
            endpoint,
            files={"file": f},
            data={"model": "whisper-1", "response_format": "text"}
        )
    
    if response.status_code == 200:
        print(f"✓ Success: {response.text}")
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")


def test_with_openai_client(audio_file: str, base_url: str = "http://localhost:8000"):
    """Test transcription using the OpenAI Python client."""
    print(f"\n{'='*60}")
    print("Testing with OpenAI Python client")
    print(f"{'='*60}")
    
    try:
        from openai import OpenAI
    except ImportError:
        print("✗ OpenAI package not installed. Install with: pip install openai")
        return
    
    # Create client pointing to local server
    client = OpenAI(
        api_key="dummy-key",  # Not used but required
        base_url=f"{base_url}/v1"
    )
    
    try:
        with open(audio_file, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        print(f"✓ Success: {transcript.text}")
    except Exception as e:
        print(f"✗ Error: {e}")


def test_health_endpoint(base_url: str = "http://localhost:8000"):
    """Test the health endpoint."""
    print(f"\n{'='*60}")
    print("Testing health endpoint")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"✓ Server is healthy: {response.json()}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python example_client.py <audio_file> [base_url]")
        print("\nExample:")
        print("  python example_client.py audio.mp3")
        print("  python example_client.py audio.mp3 http://localhost:8000")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    # Check if file exists
    if not Path(audio_file).exists():
        print(f"Error: File '{audio_file}' not found")
        sys.exit(1)
    
    print(f"\nTesting WhisperX Transcription Server")
    print(f"Server URL: {base_url}")
    print(f"Audio file: {audio_file}")
    
    # Run tests
    test_health_endpoint(base_url)
    test_with_requests(audio_file, base_url)
    test_with_openai_client(audio_file, base_url)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
