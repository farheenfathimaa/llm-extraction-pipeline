import time
import json
from enum import Enum
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser

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
        # This will now use the model name from the UI, e.g., 'gpt-4o' or 'gpt-3.5-turbo'
        return ChatOpenAI(
            model_name=self.config.get("model_name", "gpt-4o"),
            temperature=0,  # Use 0 for deterministic extraction
            openai_api_key=self.config["openai_key"],
        )

    def get_extraction_chain(self, task: ExtractionTask, custom_prompt: str = None):
        """Constructs and returns a LangChain chain for a given task using LCEL."""
        if task == ExtractionTask.CUSTOM_EXTRACTION and not custom_prompt:
            raise ValueError("Custom prompt must be provided for CUSTOM_EXTRACTION task.")

        # Create the prompt template
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
        
        # We will use the JSON output parser to ensure structured output.
        output_parser = SimpleJsonOutputParser()
        
        # This is the modern LangChain Expression Language (LCEL) syntax.
        return extraction_prompt | self.llm | output_parser

class ExtractionPipeline:
    """
    Main pipeline for document processing and LLM-based extraction.
    This class orchestrates the document processing and LLM chain execution.
    """

    def __init__(self, config: dict):
        self.config = config
        self.chain_manager = LLMChainManager(config)

    def extract_insights(
        self,
        text: str,
        extraction_type: str,
        custom_prompt: str = None,
        confidence_threshold: float = 0.7,
    ) -> dict:
        """
        Runs the extraction pipeline on a given text.
        """
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
            
            # Simple chunking strategy to prevent token limit issues
            if len(text) > 4000:
                text = text[:4000] + "..."
            
            # Use chain.invoke() as per the deprecation warning
            result = chain.invoke({"document": text})
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # The result is already a dictionary thanks to SimpleJsonOutputParser.
            return {
                "summary": result.get("summary", "No summary found."),
                "insights": result.get("insights", []),
                "confidence": 1.0,  # Placeholder, as LLMs don't directly return this
                "processing_time": processing_time
            }
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            return {
                "summary": f"Error: {str(e)}",
                "insights": [],
                "processing_time": processing_time
            }