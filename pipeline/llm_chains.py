import time
import json
from enum import Enum
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers.json import SimpleJsonOutputParser

# Import the new CacheManager
from .cache_manager import CacheManager

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
            api_key = self.config["openai_key"]
            return ChatOpenAI(
                model_name=model_name,
                temperature=0,  # Use 0 for deterministic extraction
                openai_api_key=api_key,
            )
        elif provider == "Anthropic":
            api_key = self.config["anthropic_key"]
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
        self.cache_manager = CacheManager()

    def extract_insights(
        self,
        text: str,
        extraction_type: str,
        custom_prompt: str = None,
        confidence_threshold: float = 0.7,
    ) -> dict:
        """
        Runs the extraction pipeline on a given text, with caching.
        """
        
        # Check cache first
        cached_result = self.cache_manager.get(text, extraction_type, custom_prompt)
        if cached_result:
            return cached_result
            
        start_time = time.time()
        
        try:
            task = ExtractionTask[extraction_type.upper()]
        except KeyError:
            return {
                "summary": "Error: Invalid extraction type.",
                "insights": [],
                "processing_time": 0.0
            }

        try:
            chain = self.chain_manager.get_extraction_chain(task, custom_prompt)
            
            if len(text) > 4000:
                text = text[:4000] + "..."
            
            result = chain.invoke({"document": text})
            
            # --- START: New Fallback Logic ---
            # If the output is a string (not a valid JSON object), use it as the summary
            if isinstance(result, str):
                final_result = {
                    "summary": result,
                    "insights": [],
                    "confidence": 1.0,
                    "processing_time": time.time() - start_time
                }
            # --- END: New Fallback Logic ---
            else:
                final_result = {
                    "summary": result.get("summary", "No summary found."),
                    "insights": result.get("insights", []),
                    "confidence": 1.0,
                    "processing_time": time.time() - start_time
                }
            
            # Cache the result before returning
            self.cache_manager.set(text, extraction_type, custom_prompt, final_result)
            
            return final_result
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            return {
                "summary": f"Error: {str(e)}",
                "insights": [],
                "processing_time": processing_time
            }
