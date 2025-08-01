import json
import hashlib
from pathlib import Path

class CacheManager:
    """Manages a file-based cache for LLM extraction results."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _generate_key(self, text: str, extraction_type: str, custom_prompt: str = None, config: dict = None) -> str:
        """Generates a unique hash key for a given extraction request."""
        # This key needs to be unique for each combination of inputs, including the LLM model
        key_parts = [text, extraction_type]
        if custom_prompt:
            key_parts.append(custom_prompt)
        
        # Include provider and model name from the config
        if config:
            key_parts.append(config.get("provider", ""))
            key_parts.append(config.get("model_name", ""))
            # Also include temperature if it affects output deterministically
            key_parts.append(str(config.get("temperature", 0))) 
            
        # Use a hash to create a unique file name
        hasher = hashlib.sha256()
        hasher.update("".join(key_parts).encode('utf-8'))
        return hasher.hexdigest()

    # Modified get and set methods to accept config
    def get(self, text: str, extraction_type: str, custom_prompt: str = None, config: dict = None) -> dict | None:
        """Retrieves a cached result, or None if not found."""
        key = self._generate_key(text, extraction_type, custom_prompt, config)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error reading cache file {cache_file}: {e}")
                return None
        return None

    def set(self, text: str, extraction_type: str, custom_prompt: str, data: dict, config: dict = None):
        """Saves a new result to the cache."""
        key = self._generate_key(text, extraction_type, custom_prompt, config)
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Error writing to cache file {cache_file}: {e}")