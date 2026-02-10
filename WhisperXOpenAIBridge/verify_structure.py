#!/usr/bin/env python3
"""
Verification script for WhisperXOpenAIBridge project structure
This script verifies that all required files and directories exist
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    exists = Path(filepath).is_file()
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists


def check_dir_exists(dirpath: str) -> bool:
    """Check if a directory exists"""
    exists = Path(dirpath).is_dir()
    status = "✓" if exists else "✗"
    print(f"{status} {dirpath}/")
    return exists


def main():
    """Main verification function"""
    print("=" * 60)
    print("WhisperXOpenAIBridge Project Structure Verification")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    all_checks = []
    
    print("\n📁 Core Directories:")
    all_checks.append(check_dir_exists("src"))
    all_checks.append(check_dir_exists("src/whisperx_bridge"))
    all_checks.append(check_dir_exists("tests"))
    all_checks.append(check_dir_exists("config"))
    
    print("\n📄 Configuration Files:")
    all_checks.append(check_file_exists("README.md"))
    all_checks.append(check_file_exists("pyproject.toml"))
    all_checks.append(check_file_exists("requirements.txt"))
    all_checks.append(check_file_exists(".gitignore"))
    all_checks.append(check_file_exists(".env.example"))
    
    print("\n🐍 Source Files:")
    all_checks.append(check_file_exists("src/__init__.py"))
    all_checks.append(check_file_exists("src/whisperx_bridge/__init__.py"))
    all_checks.append(check_file_exists("src/whisperx_bridge/__main__.py"))
    all_checks.append(check_file_exists("src/whisperx_bridge/config.py"))
    all_checks.append(check_file_exists("src/whisperx_bridge/server.py"))
    all_checks.append(check_file_exists("src/whisperx_bridge/transcription.py"))
    
    print("\n🧪 Test Files:")
    all_checks.append(check_file_exists("tests/__init__.py"))
    all_checks.append(check_file_exists("tests/test_config.py"))
    all_checks.append(check_file_exists("tests/test_server.py"))
    
    print("\n🐳 Docker Files:")
    all_checks.append(check_file_exists("Dockerfile"))
    all_checks.append(check_file_exists("docker-compose.yml"))
    
    print("\n📝 Example Files:")
    all_checks.append(check_file_exists("example_usage.py"))
    
    print("\n" + "=" * 60)
    
    if all(all_checks):
        print("✅ All checks passed! Project structure is complete.")
        print("\n📦 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy .env.example to .env and configure")
        print("3. Run the server: python -m src.whisperx_bridge.server")
        print("4. Run tests: pytest tests/")
        return 0
    else:
        print("❌ Some checks failed. Please review the missing files/directories.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
