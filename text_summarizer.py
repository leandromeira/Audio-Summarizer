"""
Text summarization utilities using local LLM.
"""
from pathlib import Path
from typing import Generator, Iterator
from llama_cpp import Llama

from config import (
    LLM_MODEL_PATH, N_CTX, N_GPU_LAYERS, MAX_TOKENS_LLM, 
    OUTPUT_LANGUAGE, SUMMARY_PROMPT_TEMPLATE
)


class TextSummarizer:
    """Handles text summarization using a local LLM."""
    
    def __init__(
        self, 
        model_path: Path = LLM_MODEL_PATH,
        n_ctx: int = N_CTX,
        n_gpu_layers: int = N_GPU_LAYERS,
        max_tokens: int = MAX_TOKENS_LLM,
        output_language: str = OUTPUT_LANGUAGE
    ):
        """
        Initialize the text summarizer.
        
        Args:
            model_path: Path to the LLM model file
            n_ctx: Context size for the model
            n_gpu_layers: Number of layers to run on GPU
            max_tokens: Maximum tokens for generation
            output_language: Language for the summary output
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.output_language = output_language
        self._llm = None
    
    @property
    def llm(self) -> Llama:
        """Lazy load the LLM model."""
        if self._llm is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"LLM model not found: {self.model_path}")
            
            print(f"🧠 Loading LLM model: {self.model_path.name}")
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
        return self._llm
    
    def create_prompt(self, transcript: str) -> str:
        """
        Create a prompt for summarization.
        
        Args:
            transcript: The text to summarize
            
        Returns:
            Formatted prompt string
        """
        return SUMMARY_PROMPT_TEMPLATE.format(
            transcript=transcript,
            output_language=self.output_language
        )
    
    def summarize_text(self, text: str, stream: bool = True) -> str:
        """
        Summarize the given text using the local LLM.
        
        Args:
            text: Text to summarize
            stream: Whether to stream the output
            
        Returns:
            Generated summary
        """
        prompt = self.create_prompt(text)
        
        print("🧠 Generating summary with local LLM...")
        print("\n📄 Summary output:\n")
        
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            stream=stream,
            stop=["</s>"]
        )
        
        if stream:
            return self._handle_streaming_response(response)
        else:
            return response["choices"][0]["text"].strip()
    
    def _handle_streaming_response(self, response: Iterator) -> str:
        """
        Handle streaming response from the LLM.
        
        Args:
            response: Streaming response iterator
            
        Returns:
            Complete response text
        """
        full_response = ""
        for chunk in response:
            content = chunk["choices"][0]["text"]
            print(content, end="", flush=True)
            full_response += content
        
        print("\n\n✅ Summary complete.\n")
        return full_response.strip()
