import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Config:
    """Configuration class for the LLM Information Extraction Pipeline"""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    LANGCHAIN_API_KEY: str = os.getenv('LANGCHAIN_API_KEY', '')
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    
    # Model Configuration
    DEFAULT_MODEL_PROVIDER: str = "OpenAI"
    DEFAULT_OPENAI_MODEL: str = "gpt-4-turbo"
    DEFAULT_ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
    
    # Processing Settings
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.1
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200
    
    # Evaluation Settings
    CONFIDENCE_THRESHOLD: float = 0.7
    EVALUATION_BATCH_SIZE: int = 10
    
    # File Processing
    SUPPORTED_FILE_TYPES: List[str] = ['.pdf', '.docx', '.txt', '.md']
    MAX_FILE_SIZE_MB: int = 10
    TEMP_DIR: str = "temp_uploads"
    
    # Chain Configuration
    MAX_CHAIN_LENGTH: int = 10
    PARALLEL_PROCESSING: bool = True
    VALIDATION_ENABLED: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "pipeline.log"
    
    # Prompts
    PROMPTS: Dict[str, str] = {
        "policy_conclusions": """
        Extract key policy conclusions from the following document. 
        Focus on:
        1. Main policy recommendations
        2. Evidence supporting these recommendations
        3. Implementation considerations
        4. Potential impacts or outcomes
        
        Document: {document}
        
        Please provide a structured response with clear policy conclusions.
        """,
        
        "research_insights": """
        Analyze the following research document and extract key insights:
        1. Main research findings
        2. Methodology used
        3. Implications of the findings
        4. Future research directions
        5. Limitations or constraints
        
        Document: {document}
        
        Provide a comprehensive analysis of the research insights.
        """,
        
        "summarization": """
        Provide a concise summary of the following document:
        1. Main topic and purpose
        2. Key points covered
        3. Important conclusions
        4. Relevant context
        
        Document: {document}
        
        Keep the summary informative but concise.
        """,
        
        "key_insights": """
        Extract the most important insights from this document:
        1. Novel ideas or concepts
        2. Surprising findings
        3. Practical applications
        4. Critical information
        
        Document: {document}
        
        Focus on insights that would be valuable to decision-makers.
        """,
        
        "sentiment_analysis": """
        Analyze the sentiment and tone of the following document:
        1. Overall sentiment (positive, negative, neutral)
        2. Emotional tone
        3. Author's perspective
        4. Bias detection
        
        Document: {document}
        
        Provide a balanced sentiment analysis.
        """,
        
        "entity_recognition": """
        Identify and extract key entities from the document:
        1. People and organizations
        2. Locations and places
        3. Dates and times
        4. Key concepts and terms
        5. Quantitative data
        
        Document: {document}
        
        Organize entities by category.
        """,
        
        "validation": """
        Validate the following analysis for accuracy and completeness:
        1. Check for logical consistency
        2. Verify key claims are supported
        3. Assess completeness of analysis
        4. Identify any missing elements
        
        Analysis: {analysis}
        Original Document: {document}
        
        Provide validation feedback and suggestions for improvement.
        """
    }
    
    # Evaluation Metrics
    EVALUATION_METRICS: List[str] = [
        "accuracy",
        "completeness",
        "relevance",
        "coherence",
        "processing_speed"
    ]
    
    # Chain Templates
    CHAIN_TEMPLATES: Dict[str, List[str]] = {
        "comprehensive_analysis": [
            "summarization",
            "key_insights",
            "policy_conclusions",
            "validation"
        ],
        "research_pipeline": [
            "research_insights",
            "entity_recognition",
            "sentiment_analysis",
            "validation"
        ],
        "policy_analysis": [
            "policy_conclusions",
            "key_insights",
            "validation"
        ],
        "quick_summary": [
            "summarization",
            "key_insights"
        ]
    }
    
    @classmethod
    def get_prompt(cls, prompt_type: str) -> str:
        """Get a prompt template by type"""
        return cls.PROMPTS.get(prompt_type, cls.PROMPTS["summarization"])
    
    @classmethod
    def get_chain_template(cls, template_name: str) -> List[str]:
        """Get a chain template by name"""
        return cls.CHAIN_TEMPLATES.get(template_name, cls.CHAIN_TEMPLATES["quick_summary"])
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        return {
            "openai": bool(cls.OPENAI_API_KEY),
            "langchain": bool(cls.LANGCHAIN_API_KEY),
            "anthropic": bool(cls.ANTHROPIC_API_KEY)
        }
    
    @classmethod
    def create_temp_directory(cls) -> str:
        """Create temporary directory for file processing"""
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        return cls.TEMP_DIR