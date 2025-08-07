"""
Configuration settings for the Audio Summarizer.
"""
import os
from pathlib import Path

# === DIRECTORIES ===
BASE_DIR = Path(__file__).parent
AUDIO_FOLDER = BASE_DIR / "audios"
TRANSCRIPTS_FOLDER = BASE_DIR / "transcripts"
SUMMARIES_FOLDER = BASE_DIR / "summaries"
MODELS_FOLDER = BASE_DIR / "models"

# === AUDIO PROCESSING ===
SUPPORTED_AUDIO_FORMATS = (".wav", ".mp3", ".m4a", ".aac", ".flac")
WHISPER_MODEL = "medium"
WHISPER_LANGUAGE = "pt"

# === LLM CONFIGURATION ===
LLM_MODEL_PATH = MODELS_FOLDER / "Mistral-7B-Instruct-v0.3-Q8_0.gguf"
N_CTX = 8192
N_GPU_LAYERS = 20
MAX_TOKENS_LLM = 2048
OUTPUT_LANGUAGE = "brazilian portuguese"

# === PROMPT TEMPLATE ===
SUMMARY_PROMPT_TEMPLATE = """
You are my personal learning assistant. Your role is to help me learn faster and more effectively by explaining complex topics in a clear, structured, and practical way.

Whenever I provide a class transcript, your task is to:

    1. Extract and summarize the key topics and concepts.
    2. Provide a clear, structured, and didactic summary in Markdown format.
    3. Use bullet points, organize content by topic, and avoid unnecessary jargon.
    4. Relate concepts to real-world applications whenever possible.
    5. Break down broad or complex topics into digestible parts and suggest a study path.
    6. Recommend relevant resources for further learning (books, articles, videos, exercises).
    7. Help me retain and apply the knowledge effectively.

Here is the transcript:
\"\"\"
{transcript}
\"\"\"

Please return a clean, well-organized Markdown file.
Language of the summary: {OUTPUT_LANGUAGE}.
"""
