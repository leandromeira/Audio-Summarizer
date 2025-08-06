"""
Main application logic for the Audio Summarizer.
"""
from pathlib import Path
from typing import Optional

from audio_processor import AudioProcessor
from text_summarizer import TextSummarizer
from file_manager import FileManager
from config import AUDIO_FOLDER


class AudioSummarizer:
    """Main application class that orchestrates the audio summarization process."""
    
    def __init__(self):
        """Initialize the AudioSummarizer with its components."""
        self.audio_processor = AudioProcessor()
        self.text_summarizer = TextSummarizer()
        self.file_manager = FileManager()
    
    def process_latest_audio(self) -> tuple[Path, Path]:
        """
        Process the latest audio file: transcribe and summarize.
        
        Returns:
            Tuple of (transcript_path, summary_path)
        """
        # Ensure output directories exist
        self.file_manager.ensure_directories_exist()
        
        # Find and process the latest audio file
        audio_path = self.audio_processor.find_latest_audio()
        print(f"🎵 Processing: {audio_path.name}")
        
        # Generate output file paths
        transcript_path, summary_path = self.file_manager.generate_output_paths(audio_path)
        
        # Transcribe audio
        transcript_text = self.audio_processor.transcribe_audio(audio_path)
        self.file_manager.save_file(transcript_text, transcript_path)
        
        # Generate summary
        summary_text = self.text_summarizer.summarize_text(transcript_text)
        self.file_manager.save_file(summary_text, summary_path)
        
        return transcript_path, summary_path
    
    def regenerate_summary(self, transcript_path: Path, summary_path: Path) -> None:
        """
        Regenerate summary from existing transcript.
        
        Args:
            transcript_path: Path to the transcript file
            summary_path: Path where to save the new summary
        """
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        # Read existing transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        
        # Generate new summary
        summary_text = self.text_summarizer.summarize_text(transcript_text)
        self.file_manager.save_file(summary_text, summary_path)
    
    def run_interactive(self) -> None:
        """Run the application in interactive mode with regeneration option."""
        try:
            transcript_path, summary_path = self.process_latest_audio()
            
            # Interactive loop for summary regeneration
            while True:
                regenerate = input("\n🔁 Would you like to regenerate the summary? (y/N): ").strip().lower()
                if regenerate != "y":
                    break
                
                print("\n" + "="*50)
                self.regenerate_summary(transcript_path, summary_path)
            
            print("\n🎉 Process completed successfully!")
            print(f"📄 Transcript: {transcript_path}")
            print(f"📝 Summary: {summary_path}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
        except KeyboardInterrupt:
            print("\n\n👋 Process interrupted by user.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise


def main():
    """Main entry point for the application."""
    print("🎧 Audio Summarizer Started")
    print("="*40)
    
    app = AudioSummarizer()
    app.run_interactive()


if __name__ == "__main__":
    main()
