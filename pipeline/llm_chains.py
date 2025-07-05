import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.callbacks import get_openai_callback

from config import Config

class ChainOutputParser(BaseOutputParser):
    """Custom output parser for structured responses"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into structured format"""
        try:
            # Try to parse as JSON first
            if text.strip().startswith('{') and text.strip().endswith('}'):
                return json.loads(text)
            
            # Otherwise, structure the text response
            return {
                "content": text,
                "parsed_at": datetime.now().isoformat(),
                "format": "text"
            }
        except json.JSONDecodeError:
            return {
                "content": text,
                "parsed_at": datetime.now().isoformat(),
                "format": "text",
                "error": "Failed to parse as JSON"
            }

class LLMChainManager:
    """Manages multiple LLM chains for different extraction tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.chains = {}
        self.llm = None
        self.output_parser = ChainOutputParser()
        
        # Initialize LLM
        self._initialize_llm()
        
        # Create chains
        self._create_chains()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        try:
            if self.config.LLM_PROVIDER == "openai":
                self.llm = ChatOpenAI(
                    model_name=self.config.LLM_MODEL,
                    temperature=self.config.LLM_TEMPERATURE,
                    openai_api_key=self.config.OPENAI_API_KEY,
                    max_tokens=self.config.MAX_TOKENS
                )
            elif self.config.LLM_PROVIDER == "anthropic":
                # Note: You'll need to add anthropic integration
                # from langchain.llms import Anthropic
                # self.llm = Anthropic(...)
                raise NotImplementedError("Anthropic integration not implemented yet")
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.LLM_PROVIDER}")
                
            self.logger.info(f"Initialized {self.config.LLM_PROVIDER} LLM")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _create_chains(self):
        """Create different chains for various extraction tasks"""
        # Summarization chain
        self.chains['summarization'] = self._create_summarization_chain()
        
        # Key insights extraction chain
        self.chains['key_insights'] = self._create_key_insights_chain()
        
        # Policy conclusions chain
        self.chains['policy_conclusions'] = self._create_policy_conclusions_chain()
        
        # Validation chain
        self.chains['validation'] = self._create_validation_chain()
        
        # Custom analysis chain
        self.chains['custom_analysis'] = self._create_custom_analysis_chain()
    
    def _create_summarization_chain(self) -> LLMChain:
        """Create summarization chain"""
        prompt = PromptTemplate(
            input_variables=["text", "max_length"],
            template="""
            Please provide a comprehensive summary of the following text.
            
            Requirements:
            - Maximum length: {max_length} words
            - Include key points and main themes
            - Maintain objectivity and accuracy
            - Use clear, professional language
            
            Text to summarize:
            {text}
            
            Summary:
            """
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser,
            verbose=True
        )
    
    def _create_key_insights_chain(self) -> LLMChain:
        """Create key insights extraction chain"""
        prompt = PromptTemplate(
            input_variables=["text", "focus_areas"],
            template="""
            Extract key insights from the following text, focusing on the specified areas.
            
            Focus Areas: {focus_areas}
            
            Please provide insights in the following JSON format:
            {{
                "insights": [
                    {{
                        "category": "category_name",
                        "insight": "detailed insight",
                        "evidence": "supporting evidence from text",
                        "confidence": "high/medium/low"
                    }}
                ],
                "themes": ["theme1", "theme2", "theme3"],
                "recommendations": ["recommendation1", "recommendation2"]
            }}
            
            Text to analyze:
            {text}
            
            Key Insights:
            """
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser,
            verbose=True
        )
    
    def _create_policy_conclusions_chain(self) -> LLMChain:
        """Create policy conclusions extraction chain"""
        prompt = PromptTemplate(
            input_variables=["text", "policy_context"],
            template="""
            Analyze the following text and extract policy-relevant conclusions.
            
            Policy Context: {policy_context}
            
            Please provide conclusions in the following JSON format:
            {{
                "policy_conclusions": [
                    {{
                        "conclusion": "specific policy conclusion",
                        "rationale": "reasoning behind the conclusion",
                        "implications": "potential policy implications",
                        "evidence_strength": "strong/moderate/weak"
                    }}
                ],
                "implementation_considerations": ["consideration1", "consideration2"],
                "potential_challenges": ["challenge1", "challenge2"],
                "success_factors": ["factor1", "factor2"]
            }}
            
            Text to analyze:
            {text}
            
            Policy Conclusions:
            """
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser,
            verbose=True
        )
    
    def _create_validation_chain(self) -> LLMChain:
        """Create validation chain to check previous outputs"""
        prompt = PromptTemplate(
            input_variables=["original_text", "extracted_info", "validation_criteria"],
            template="""
            Validate the following extracted information against the original text.
            
            Validation Criteria: {validation_criteria}
            
            Original Text:
            {original_text}
            
            Extracted Information:
            {extracted_info}
            
            Please provide validation results in the following JSON format:
            {{
                "validation_results": [
                    {{
                        "item": "item being validated",
                        "is_accurate": true/false,
                        "confidence": "high/medium/low",
                        "issues": ["issue1", "issue2"] or [],
                        "suggestions": ["suggestion1", "suggestion2"] or []
                    }}
                ],
                "overall_accuracy": "high/medium/low",
                "reliability_score": 0.85,
                "validation_notes": "additional notes"
            }}
            
            Validation Results:
            """
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser,
            verbose=True
        )
    
    def _create_custom_analysis_chain(self) -> LLMChain:
        """Create custom analysis chain for user-defined prompts"""
        prompt = PromptTemplate(
            input_variables=["text", "custom_prompt", "output_format"],
            template="""
            {custom_prompt}
            
            Output Format: {output_format}
            
            Text to analyze:
            {text}
            
            Analysis:
            """
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.output_parser,
            verbose=True
        )
    
    def run_chain(self, chain_name: str, **kwargs) -> Dict[str, Any]:
        """Run a specific chain with given parameters"""
        if chain_name not in self.chains:
            raise ValueError(f"Chain '{chain_name}' not found")
        
        try:
            with get_openai_callback() as cb:
                result = self.chains[chain_name].run(**kwargs)
                
                # Add metadata
                if isinstance(result, dict):
                    result['metadata'] = {
                        'chain_name': chain_name,
                        'timestamp': datetime.now().isoformat(),
                        'tokens_used': cb.total_tokens,
                        'cost': cb.total_cost
                    }
                else:
                    result = {
                        'content': result,
                        'metadata': {
                            'chain_name': chain_name,
                            'timestamp': datetime.now().isoformat(),
                            'tokens_used': cb.total_tokens,
                            'cost': cb.total_cost
                        }
                    }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error running chain '{chain_name}': {str(e)}")
            return {
                'error': str(e),
                'chain_name': chain_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_pipeline(self, text: str, chain_sequence: List[str], **kwargs) -> Dict[str, Any]:
        """Run multiple chains in sequence"""
        results = {}
        intermediate_results = {}
        
        for chain_name in chain_sequence:
            try:
                # Prepare inputs for current chain
                chain_inputs = self._prepare_chain_inputs(
                    chain_name, text, intermediate_results, **kwargs
                )
                
                # Run chain
                result = self.run_chain(chain_name, **chain_inputs)
                results[chain_name] = result
                
                # Store for next chain
                intermediate_results[chain_name] = result
                
                self.logger.info(f"Completed chain: {chain_name}")
                
            except Exception as e:
                self.logger.error(f"Pipeline failed at chain '{chain_name}': {str(e)}")
                results[chain_name] = {'error': str(e)}
                break
        
        return {
            'pipeline_results': results,
            'status': 'completed' if len(results) == len(chain_sequence) else 'partial',
            'timestamp': datetime.now().isoformat()
        }
    
    def _prepare_chain_inputs(self, chain_name: str, text: str, 
                            intermediate_results: Dict, **kwargs) -> Dict[str, Any]:
        """Prepare inputs for a specific chain"""
        inputs = {'text': text}
        
        if chain_name == 'summarization':
            inputs['max_length'] = kwargs.get('max_length', 500)
            
        elif chain_name == 'key_insights':
            inputs['focus_areas'] = kwargs.get('focus_areas', 'general analysis')
            
        elif chain_name == 'policy_conclusions':
            inputs['policy_context'] = kwargs.get('policy_context', 'general policy analysis')
            
        elif chain_name == 'validation':
            inputs['original_text'] = text
            inputs['extracted_info'] = json.dumps(intermediate_results)
            inputs['validation_criteria'] = kwargs.get('validation_criteria', 'accuracy and completeness')
            
        elif chain_name == 'custom_analysis':
            inputs['custom_prompt'] = kwargs.get('custom_prompt', 'Analyze the following text:')
            inputs['output_format'] = kwargs.get('output_format', 'structured analysis')
        
        return inputs
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about available chains"""
        return {
            'available_chains': list(self.chains.keys()),
            'llm_provider': self.config.LLM_PROVIDER,
            'llm_model': self.config.LLM_MODEL,
            'predefined_templates': list(self.config.CHAIN_TEMPLATES.keys())
        }
    
    def estimate_cost(self, text: str, chain_sequence: List[str]) -> Dict[str, Any]:
        """Estimate the cost of running a pipeline"""
        # Rough estimation based on token count
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.LLM_MODEL)
            input_tokens = len(encoding.encode(text))
            
            # Estimate output tokens (rough approximation)
            estimated_output_tokens = input_tokens * 0.3 * len(chain_sequence)
            
            # Cost estimation (these are approximate rates)
            cost_per_1k_input = 0.001  # $0.001 per 1K input tokens
            cost_per_1k_output = 0.002  # $0.002 per 1K output tokens
            
            estimated_cost = (input_tokens / 1000 * cost_per_1k_input + 
                            estimated_output_tokens / 1000 * cost_per_1k_output)
            
            return {
                'estimated_input_tokens': input_tokens,
                'estimated_output_tokens': int(estimated_output_tokens),
                'estimated_total_tokens': int(input_tokens + estimated_output_tokens),
                'estimated_cost': round(estimated_cost, 4),
                'chain_sequence': chain_sequence
            }
            
        except Exception as e:
            return {
                'error': f"Could not estimate cost: {str(e)}",
                'chain_sequence': chain_sequence
            }