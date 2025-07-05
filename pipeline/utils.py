import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

class FileManager:
    """Utility class for file operations and management"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.upload_dir = self.base_path / "uploads"
        self.output_dir = self.base_path / "outputs"
        self.cache_dir = self.base_path / "cache"
        self.logs_dir = self.base_path / "logs"
        
        for dir_path in [self.upload_dir, self.output_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file and return path"""
        file_path = self.upload_dir / filename
        with open(file_path, 'wb') as f:
            f.write(file_content)
        return str(file_path)
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save pipeline results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_results_{timestamp}.json"
        
        file_path = self.output_dir / filename
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return str(file_path)
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load pipeline results from JSON file"""
        file_path = self.output_dir / filename
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_cache_key(self, content: str, config: Dict[str, Any]) -> str:
        """Generate cache key for content and configuration"""
        cache_string = content + json.dumps(config, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def save_to_cache(self, key: str, data: Any) -> None:
        """Save data to cache"""
        cache_path = self.cache_dir / f"{key}.json"
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_from_cache(self, key: str) -> Optional[Any]:
        """Load data from cache"""
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def clear_cache(self) -> None:
        """Clear all cache files"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
    
    def get_uploaded_files(self) -> List[str]:
        """Get list of uploaded files"""
        return [f.name for f in self.upload_dir.iterdir() if f.is_file()]
    
    def get_output_files(self) -> List[str]:
        """Get list of output files"""
        return [f.name for f in self.output_dir.iterdir() if f.is_file()]

class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        import re
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\]', '', text)
        
        return text.strip()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_sentence = chunk.rfind('.')
                if last_sentence > chunk_size * 0.8:  # If we found a sentence boundary in the last 20%
                    chunk = chunk[:last_sentence + 1]
                    end = start + len(chunk)
            
            chunks.append(chunk)
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text"""
        from collections import Counter
        import re
        
        # Simple keyword extraction (you can enhance this with NLP libraries)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'that', 'with', 'have', 'this', 'will', 'been', 'from', 'they', 'know',
            'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come',
            'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take',
            'than', 'them', 'well', 'were', 'your', 'more', 'also', 'back', 'other',
            'into', 'after', 'first', 'never', 'these', 'think', 'where', 'being',
            'every', 'great', 'might', 'shall', 'still', 'those', 'under', 'while'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(top_k)]
    
    @staticmethod
    def calculate_readability(text: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        import re
        
        # Count sentences, words, and syllables
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        
        # Simple syllable counting
        syllables = 0
        for word in text.split():
            word = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            prev_char_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_char_was_vowel:
                        syllable_count += 1
                    prev_char_was_vowel = True
                else:
                    prev_char_was_vowel = False
            
            if word.endswith('e'):
                syllable_count -= 1
            if syllable_count == 0:
                syllable_count = 1
            
            syllables += syllable_count
        
        # Flesch Reading Ease
        if sentences > 0 and words > 0:
            flesch_score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        else:
            flesch_score = 0
        
        # Flesch-Kincaid Grade Level
        if sentences > 0 and words > 0:
            fk_grade = (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
        else:
            fk_grade = 0
        
        return {
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'flesch_kincaid_grade': max(0, fk_grade),
            'words': words,
            'sentences': sentences,
            'syllables': syllables
        }

class Logger:
    """Utility class for logging operations"""
    
    def __init__(self, name: str = "pipeline", log_dir: str = "data/logs"):
        self.logger = logging.getLogger(name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

class DataExporter:
    """Utility class for exporting data in various formats"""
    
    @staticmethod
    def to_csv(data: List[Dict[str, Any]], filename: str) -> str:
        """Export data to CSV"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    @staticmethod
    def to_excel(data: Dict[str, List[Dict[str, Any]]], filename: str) -> str:
        """Export data to Excel with multiple sheets"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, sheet_data in data.items():
                df = pd.DataFrame(sheet_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return filename
    
    @staticmethod
    def to_json(data: Any, filename: str) -> str:
        """Export data to JSON"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filename
    
    @staticmethod
    def to_markdown(data: Dict[str, Any], filename: str) -> str:
        """Export data to Markdown report"""
        with open(filename, 'w') as f:
            f.write("# Pipeline Results Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for section, content in data.items():
                f.write(f"## {section.replace('_', ' ').title()}\n\n")
                
                if isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"**{key}:** {value}\n\n")
                elif isinstance(content, list):
                    for item in content:
                        f.write(f"- {item}\n")
                    f.write("\n")
                else:
                    f.write(f"{content}\n\n")
        
        return filename

def validate_api_keys() -> Dict[str, bool]:
    """Validate API keys"""
    api_keys = {
        'openai': bool(os.getenv('OPENAI_API_KEY')),
        'anthropic': bool(os.getenv('ANTHROPIC_API_KEY')),
        'huggingface': bool(os.getenv('HUGGINGFACE_API_KEY'))
    }
    return api_keys

def estimate_tokens(text: str) -> int:
    """Estimate token count for text"""
    # Rough estimation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def calculate_cost(tokens: int, model: str = "gpt-3.5-turbo") -> float:
    """Calculate estimated cost for API calls"""
    # Pricing as of 2024 (update as needed)
    pricing = {
        "gpt-3.5-turbo": 0.0015 / 1000,  # per 1K tokens
        "gpt-4": 0.03 / 1000,
        "gpt-4-turbo": 0.01 / 1000,
        "claude-3-haiku": 0.00025 / 1000,
        "claude-3-sonnet": 0.003 / 1000,
        "claude-3-opus": 0.015 / 1000
    }
    
    return tokens * pricing.get(model, 0.002 / 1000)

def format_results_for_display(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format results for better display in Streamlit"""
    formatted = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            formatted[key] = format_results_for_display(value)
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict):
                formatted[key] = pd.DataFrame(value)
            else:
                formatted[key] = value
        else:
            formatted[key] = value
    
    return formatted