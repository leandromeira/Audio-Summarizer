# 🎧 Audio Summarizer

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-green.svg)](https://github.com/openai/whisper)
[![LLM](https://img.shields.io/badge/LLM-Mistral_7B-purple.svg)](https://mistral.ai/)

An intelligent tool for transcribing and summarizing audio files using Whisper for transcription and a local LLM (Mistral) for summarization.

---

## 🧠 Features

- **Automatic audio transcription** using OpenAI Whisper (`medium` model)
- **Intelligent summarization** with local LLM (Mistral 7B Instruct)
- **Structured outputs** in Markdown format
- **Educational summaries** with key concepts, practical applications, study suggestions, and recommended resources
- **Automatic processing** of the latest audio file
- **Summary regeneration** option
- **Flexible configuration** of model parameters

---

## 🖥️ System Requirements

- **Operating System:** Linux (tested on Ubuntu)
- **RAM:** 16 GB minimum (32 GB recommended)
- **GPU:** NVIDIA (optional, for Whisper and LLM acceleration)
- **Disk Space:** ~8 GB (models + audio files)
- **Python:** 3.10 or higher

### 📋 Current Project Configuration

- **Whisper Model:** `medium` (Portuguese language)
- **LLM Model:** `Mistral-7B-Instruct-v0.3-Q8_0.gguf`
- **LLM Context:** 8192 tokens
- **GPU Layers:** 20 (adjust according to your GPU)
- **Max Tokens:** 2048 per summary
- **Output Language:** Brazilian Portuguese

---

## 📦 Installation Guide

### 1. Install Python and System Dependencies

Make sure you have Python 3.10+ installed:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv cmake build-essential libopenblas-dev
```

### 2. Create Virtual Environment

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
```

### 3. Install Python Packages

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install packages individually
pip install openai-whisper torch llama-cpp-python

# For NVIDIA GPU (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download Models

#### 🧠 Whisper Model

The `medium` model will be downloaded automatically on first run.

#### 📚 LLM Models (Mistral 7B Instruct)

Download one of the GGUF models from Hugging Face and place it in the `models/` folder:

**Option 1: Q8_0 (best quality, heavier)**

```bash
wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf
```

**Option 2: Q5_K_M (balanced)**

```bash
wget -P models/ https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf
```

**Option 3: Mistral 7B v0.3 (newest)**

```bash
wget -P models/ https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q8_0.gguf
```

> **Note:** Adjust the `LLM_MODEL_PATH` in `transcribe_and_summarize.py` if using a different model.

---

## 🚀 How to Use

### 1. Folder Structure

```
Audio Summarizer/
├── audios/                     # Place your audio files here
│   └── audio_file.wav
├── transcripts/                # Auto-generated transcriptions
│   └── audio_file_transcript.txt
├── summaries/                  # Markdown summaries
│   └── audio_file_summary.md
├── models/                     # Downloaded models
│   └── Mistral-7B-Instruct-v0.3-Q8_0.gguf
├── main.py                     # Main application entry point
├── config.py                   # Configuration settings
├── audio_processor.py          # Audio transcription logic
├── text_summarizer.py          # Text summarization logic
├── file_manager.py             # File handling utilities
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── run.sh                      # Execution script
└── README.md                   # This file
```

### 2. Prepare Audio Files

Place your audio files in the `audios/` folder. Supported formats:

- `.mp3`
- `.wav`
- `.m4a`
- `.aac`
- `.flac`

> **Tip:** The script automatically processes the most recent file (by modification date).

### 3. Run the Script

```bash
./run.sh
```

**What happens:**

1. 🔊 **Transcription:** Converts the latest audio to text using Whisper
2. 💾 **Save:** Saves transcription to `transcripts/`
3. 🧠 **Summarization:** Generates structured summary with local LLM
4. 📝 **Output:** Saves Markdown summary to `summaries/`
5. 🔁 **Option:** Asks if you want to regenerate the summary

### 4. Customize Settings

Edit variables in `config.py`:

```python
# Main configurations
WHISPER_MODEL = "medium"          # small, base, medium, large
LLM_MODEL_PATH = "models/..."     # Path to your model
N_CTX = 8192                      # LLM context
N_GPU_LAYERS = 20                 # Adjust according to your GPU
MAX_TOKENS_LLM = 2048            # Max tokens for summary
OUTPUT_LANGUAGE = "brazilian portuguese"  # Output language
```

## 🔧 Troubleshooting

### Common Issues

**1. Error installing `llama-cpp-python`:**

```bash
# Install system dependencies
sudo apt install cmake build-essential libopenblas-dev

# Reinstall package
pip uninstall llama-cpp-python
pip install llama-cpp-python
```

**2. LLM model not found:**

- Check if file is in `models/` folder
- Confirm filename in `LLM_MODEL_PATH` variable

**3. Whisper too slow:**

- Consider using smaller model: `small` or `base`
- Install PyTorch with CUDA support for GPU acceleration

**4. Out of memory (RAM/VRAM):**

- Reduce `N_CTX` to 4096 or 2048
- Decrease `N_GPU_LAYERS` or use `N_GPU_LAYERS = 0` (CPU only)
- Use smaller model (Q5_K_M instead of Q8_0)

**5. Summary in Portuguese instead of English:**

- Check that `OUTPUT_LANGUAGE = "english"` is correct
- The fix has already been applied to the code

### Performance Optimizations

**For NVIDIA GPUs:**

```python
N_GPU_LAYERS = 30  # Increase to use more GPU
```

**For low-RAM systems:**

```python
N_CTX = 2048       # Decrease context
MAX_TOKENS_LLM = 1024  # Decrease output tokens
```

---

## 📚 Use Cases

- ✅ **Review recorded classes** efficiently
- ✅ **Generate structured notes** for revision
- ✅ **Create personal knowledge base** from content
- ✅ **Prepare for presentations** and exams
- ✅ **Guided study** with personalized roadmaps

---

---

## 🏗️ Project Architecture

The project follows a modular architecture with clear separation of concerns:

### Core Components

- **AudioProcessor**: Handles audio file discovery and transcription using Whisper
- **TextSummarizer**: Manages text summarization using local LLM (Mistral)
- **FileManager**: Handles all file operations (reading, writing, path management)
- **AudioSummarizer**: Main orchestrator that coordinates all components

### Design Principles

- **Single Responsibility**: Each class has a specific, well-defined purpose
- **Dependency Injection**: Components can be easily swapped or configured
- **Lazy Loading**: Resources are loaded only when needed
- **Error Handling**: Comprehensive error handling with informative messages
- **Type Safety**: Full type hints for better IDE support and documentation

### Configuration Management

All settings are centralized in `config.py`:

- Model paths and parameters
- Directory structures
- Language settings
- Prompt templates

## 🔧 Advanced Customization

### Modify LLM Prompt

Edit the `SUMMARY_PROMPT_TEMPLATE` variable in `config.py` to customize:

- Summary style
- Specific focus (more technical, more practical, etc.)
- Output format
- Language and tone

### Advanced Usage

The modular architecture allows you to use individual components programmatically:

```python
from audio_processor import AudioProcessor
from text_summarizer import TextSummarizer
from file_manager import FileManager

# Initialize components
audio_processor = AudioProcessor(model_name="small")
text_summarizer = TextSummarizer(output_language="english")
file_manager = FileManager()

# Process specific file
audio_path = Path("audios/my_file.wav")
transcript = audio_processor.transcribe_audio(audio_path)
summary = text_summarizer.summarize_text(transcript)
```

## ✅ License and Legal Considerations

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Important:**

- ✅ Ensure you have permission to record and process content
- ✅ Respect course material terms of use
- ✅ Use only for personal study purposes
- ❌ Do not redistribute copyrighted content

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/audio-summarizer.git
   cd audio-summarizer
   ```
3. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Make your changes**
6. **Test your changes**:
   ```bash
   ./run.sh            # Test main application
   ```
7. **Submit a pull request**

### Development Areas

- Support for more audio formats
- Graphical user interface (GUI)
- Integration with other study tools
- LLM prompt improvements
- Batch processing support
- Docker containerization
- Web interface
- Cloud deployment options

### Code Style

- Follow PEP 8 standards
- Use type hints
- Add docstrings to all functions and classes
- Include unit tests for new features

---

## 📞 Support

If you encounter issues:

1. Check the **Troubleshooting** section
2. Confirm all requirements are installed
3. Test with a small audio file first
4. Open an issue on the project repository

---

## 📈 Version History

- **v2.0.0** - Complete refactor with modular architecture
- **v1.0.0** - Initial monolithic implementation

---

**Happy learning! 📘✨**
