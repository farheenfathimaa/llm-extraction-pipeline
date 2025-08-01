import time
import json
from enum import Enum
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers.json import SimpleJsonOutputParser

# Import the new CacheManager
from .cache_manager import CacheManager

# Import specific exceptions for better error handling
import openai
from anthropic import RateLimitError, APIStatusError # Corrected import for Anthropic exceptions
import logging

logger = logging.getLogger(__name__) # Initialize a logger for this module

class ExtractionTask(Enum):
    """Enumeration for different extraction tasks with default prompts."""
    POLICY_CONCLUSIONS = {
        "prompt": "Extract the key policy conclusions from the document. Format as a JSON object with a 'summary' and a list of 'insights'."
    }
    RESEARCH_INSIGHTS = {
        "prompt": "Extract the main research insights and findings from the document. Format as a JSON object with a 'summary' and a list of 'insights'."
    }
    KEY_FINDINGS = {
        "prompt": "Identify and summarize the most important findings presented in the text. Format as a JSON object with a 'summary' and a list of 'insights'."
    }
    RECOMMENDATIONS = {
        "prompt": "List all the recommendations and suggestions made in the document. Format as a JSON object with a 'summary' and a list of 'insights'."
    }
    CUSTOM_EXTRACTION = {
        "prompt": "{custom_prompt}"
    }

class LLMChainManager:
    """Manages the LLM chain configuration and invocation."""

    def __init__(self, config: dict):
        self.config = config
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initializes the LLM model from the configuration."""
        provider = self.config.get("provider", "OpenAI")
        model_name = self.config.get("model_name", "gpt-4o")
        
        if provider == "OpenAI":
            api_key = self.config.get("openai_key") # Use .get() for safer access
            if not api_key:
                logger.error("OpenAI API key is not provided in configuration.")
                raise ValueError("OpenAI API key is missing.")
            return ChatOpenAI(
                model_name=model_name,
                temperature=0,  # Use 0 for deterministic extraction
                openai_api_key=api_key,
            )
        elif provider == "Anthropic":
            api_key = self.config.get("anthropic_key") # Use .get() for safer access
            if not api_key:
                logger.error("Anthropic API key is not provided in configuration.")
                raise ValueError("Anthropic API key is missing.")
            return ChatAnthropic(
                model_name=model_name,
                temperature=0,
                anthropic_api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")


    def get_extraction_chain(self, task: ExtractionTask, custom_prompt: str = None):
        """Constructs and returns a LangChain chain for a given task using LCEL."""
        if task == ExtractionTask.CUSTOM_EXTRACTION and not custom_prompt:
            raise ValueError("Custom prompt must be provided for CUSTOM_EXTRACTION task.")

        prompt_str = task.value["prompt"]
        if task == ExtractionTask.CUSTOM_EXTRACTION:
            prompt_str = prompt_str.format(custom_prompt=custom_prompt)
        
        # Add a clear instruction for JSON output
        final_prompt_template = (
            "You are an expert at extracting information. "
            "Your task is to extract the requested information from the provided document. "
            "Your output MUST be a valid JSON object. Do not include any other text.\n\n"
            f"{prompt_str}\n\n"
            "Document: ```{document}```"
        )
        extraction_prompt = PromptTemplate.from_template(final_prompt_template)
        
        output_parser = SimpleJsonOutputParser()
        
        return extraction_prompt | self.llm | output_parser

class ExtractionPipeline:
    """
    Main pipeline for document processing and LLM-based extraction.
    This class orchestrates the document processing and LLM chain execution.
    """

    def __init__(self, config: dict):
        self.config = config
        self.chain_manager = LLMChainManager(config)
        self.cache_manager = CacheManager() # This refers to pipeline/cache_manager.py's CacheManager

    def extract_insights(
        self,
        text: str,
        extraction_type: str,
        custom_prompt: str = None,
        confidence_threshold: float = 0.7, # This threshold isn't used in the current LLM call logic
        max_retries: int = 3, # New parameter for retries
        retry_delay: int = 2 # New parameter for retry delay
    ) -> dict:
        """
        Runs the extraction pipeline on a given text, with caching and improved error handling.
        """
        
        # Check cache first
        # Ensure the config is passed to generate_key for model-specific caching
        cached_result = self.cache_manager.get(text, extraction_type, custom_prompt, config=self.config)
        if cached_result:
            return cached_result
            
        start_time = time.time()
        
        try:
            task = ExtractionTask[extraction_type.upper()]
        except KeyError:
            return {
                "summary": f"Error: Invalid extraction type '{extraction_type}'. Please choose from: {', '.join([e.name for e in ExtractionTask])}.",
                "insights": [],
                "processing_time": 0.0,
                "error": "InvalidExtractionType"
            }

        retries = 0
        while retries <= max_retries:
            try:
                chain = self.chain_manager.get_extraction_chain(task, custom_prompt)
                
                # Trim text if too long for initial call, as per existing logic in app.py
                # Note: Langchain models usually handle chunking internally or you'd pass smaller chunks.
                # This trim might cut off important context, consider DocumentProcessor.chunk_text instead
                if len(text) > 4000:
                    logger.warning(f"Document length ({len(text)}) exceeds 4000 characters. Truncating for LLM call.")
                    text_for_llm = text[:4000] + "..."
                else:
                    text_for_llm = text

                result = chain.invoke({"document": text_for_llm})
                
                # If the output is a string (not a valid JSON object), use it as the summary
                if isinstance(result, str):
                    final_result = {
                        "summary": result,
                        "insights": [],
                        "confidence": 1.0, # Placeholder, as LLMs don't typically return confidence directly
                        "processing_time": time.time() - start_time
                    }
                else:
                    final_result = {
                        "summary": result.get("summary", "No summary found."),
                        "insights": result.get("insights", []),
                        "confidence": 1.0, # Placeholder
                        "processing_time": time.time() - start_time
                    }
                
                # Cache the result before returning
                self.cache_manager.set(text, extraction_type, custom_prompt, final_result, config=self.config) # Pass config
                
                return final_result

            except (openai.RateLimitError, RateLimitError) as e: # Use RateLimitError directly now
                # Handle rate limit errors with retry
                if retries < max_retries:
                    wait_time = retry_delay * (2 ** retries) # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {wait_time:.2f} seconds... (Attempt {retries + 1}/{max_retries}) Error: {e}")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logger.error(f"Max retries reached for RateLimitError: {e}")
                    return {
                        "summary": f"Error: API rate limit exceeded after {max_retries} retries. Please wait and try again. Details: {str(e)}",
                        "insights": [],
                        "processing_time": time.time() - start_time,
                        "error": "RateLimitExceeded"
                    }
            except (openai.APIStatusError, APIStatusError) as e: # Use APIStatusError directly now
                # Catch API errors, including insufficient_quota
                error_message = str(e)
                if "insufficient_quota" in error_message or "billing" in error_message:
                    logger.error(f"Quota Error: {error_message}")
                    return {
                        "summary": f"Error: LLM API Quota Exceeded. Please check your billing details or API key for {self.config.get('provider')}. Details: {error_message}",
                        "insights": [],
                        "processing_time": time.time() - start_time,
                        "error": "InsufficientQuota"
                    }
                else:
                    logger.error(f"LLM API Error: {error_message}")
                    return {
                        "summary": f"Error: An LLM API error occurred. Details: {error_message}",
                        "insights": [],
                        "processing_time": time.time() - start_time,
                        "error": "LLMAPIError"
                    }
            except Exception as e:
                # Catch any other unexpected errors
                logger.error(f"An unexpected error occurred during LLM extraction: {e}")
                return {
                    "summary": f"Error: An unexpected error occurred during extraction: {str(e)}",
                    "insights": [],
                    "processing_time": time.time() - start_time,
                    "error": "UnexpectedError"
                }
        # This part should ideally not be reached if retries are handled correctly
        return {
            "summary": "Error: Unknown issue after retries. Check logs.",
            "insights": [],
            "processing_time": time.time() - start_time,
            "error": "Unknown"
        }