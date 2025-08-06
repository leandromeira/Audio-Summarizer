"""
Audio processing utilities for transcription.
"""
import os
from pathlib import Path
from typing import Optional
import whisper

from config import AUDIO_FOLDER, SUPPORTED_AUDIO_FORMATS, WHISPER_MODEL, WHISPER_LANGUAGE


class AudioProcessor:
    """Handles audio file processing and transcription."""
    
    def __init__(self, model_name: str = WHISPER_MODEL):
        """Initialize the audio processor with a Whisper model."""
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            print(f"🔊 Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name)
        return self._model
    
    def find_latest_audio(self, folder: Path = AUDIO_FOLDER) -> Path:
        """
        Find the most recently modified audio file in the specified folder.
        
        Args:
            folder: Path to the folder containing audio files
            
        Returns:
            Path to the latest audio file
            
        Raises:
            FileNotFoundError: If no audio files are found
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Audio folder '{folder}' does not exist")
        
        audio_files = [
            f for f in folder.iterdir() 
            if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_FORMATS
        ]
        
        if not audio_files:
            raise FileNotFoundError(
                f"No audio files found in '{folder}'. "
                f"Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            )
        
        # Sort by modification time, most recent first
        latest_file = max(audio_files, key=lambda f: f.stat().st_mtime)
        return latest_file
    
    def transcribe_audio(self, audio_path: Path, language: str = WHISPER_LANGUAGE) -> str:
        """
        Transcribe audio file to text using Whisper.
        
        Args:
            audio_path: Path to the audio file
            language: Language code for transcription
            
        Returns:
            Transcribed text
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"🔊 Transcribing audio: {audio_path.name}")
        result = self.model.transcribe(str(audio_path), language=language)
        return result["text"]
