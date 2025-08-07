"""
Audio Summarizer

An intelligent tool for transcribing and summarizing audio files using 
Whisper for transcription and a local LLM for summarization.

This is a standalone application, not a pip-installable package.
Run with: ./run.sh or python3 main.py

Author: Leandro Meira Marinho Queiróz
License: MIT
Version: 2.0.1
"""

__version__ = "2.0.1"
__author__ = "Leandro Meira Marinho Queiróz"
__license__ = "MIT"
__description__ = "Intelligent audio transcription and summarization tool (standalone)"

from .main import AudioSummarizer, main
from .audio_processor import AudioProcessor
from .text_summarizer import TextSummarizer
from .file_manager import FileManager

__all__ = [
    "AudioSummarizer",
    "AudioProcessor", 
    "TextSummarizer",
    "FileManager",
    "main"
]
