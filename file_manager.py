"""
File handling utilities for the Audio Summarizer.
"""
import os
from pathlib import Path
from typing import Union

from config import TRANSCRIPTS_FOLDER, SUMMARIES_FOLDER


class FileManager:
    """Handles file operations for transcripts and summaries."""
    
    @staticmethod
    def ensure_directories_exist():
        """Create necessary directories if they don't exist."""
        directories = [TRANSCRIPTS_FOLDER, SUMMARIES_FOLDER]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Directory ready: {directory}")
    
    @staticmethod
    def save_file(content: str, file_path: Union[str, Path], encoding: str = "utf-8") -> None:
        """
        Save content to a file.
        
        Args:
            content: Text content to save
            file_path: Path where to save the file
            encoding: File encoding (default: utf-8)
        """
        file_path = Path(file_path)
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)
            print(f"💾 File saved: {file_path}")
        except Exception as e:
            print(f"❌ Error saving file {file_path}: {e}")
            raise
    
    @staticmethod
    def generate_output_paths(audio_file_path: Path) -> tuple[Path, Path]:
        """
        Generate paths for transcript and summary files based on audio filename.
        
        Args:
            audio_file_path: Path to the original audio file
            
        Returns:
            Tuple of (transcript_path, summary_path)
        """
        base_name = audio_file_path.stem  # filename without extension
        
        transcript_path = TRANSCRIPTS_FOLDER / f"{base_name}_transcript.txt"
        summary_path = SUMMARIES_FOLDER / f"{base_name}_summary.md"
        
        return transcript_path, summary_path
    
    @staticmethod
    def file_exists(file_path: Union[str, Path]) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        return Path(file_path).exists()
