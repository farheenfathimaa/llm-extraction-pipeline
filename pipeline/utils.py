import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# File, cache, and logging helpers
# ---------------------------------------------------------------------------

class FileManager:
    """Utility class for file operations, uploads, outputs, and simple caching."""
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        self.upload_dir = self.base_path / "uploads"
        self.output_dir = self.base_path / "outputs"
        self.cache_dir = self.base_path / "cache"
        self.logs_dir = self.base_path / "logs"

        for path in (self.upload_dir, self.output_dir, self.cache_dir, self.logs_dir):
            path.mkdir(exist_ok=True)

    # ---------- uploads / outputs ----------
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        out = self.upload_dir / filename
        out.write_bytes(file_content)
        return str(out)

    def save_results(self, results: Dict[str, Any], filename: str | None = None) -> str:
        filename = (
            filename
            or f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        out = self.output_dir / filename
        out.write_text(json.dumps(results, indent=2, default=str))
        return str(out)

    def load_results(self, filename: str) -> Dict[str, Any]:
        return json.loads((self.output_dir / filename).read_text())

    # ---------- cache ----------
    @staticmethod
    def _cache_key(content: str, config: Dict[str, Any]) -> str:
        return hashlib.md5((content + json.dumps(config, sort_keys=True)).encode()).hexdigest()

    def save_to_cache(self, key: str, data: Any) -> None:
        (self.cache_dir / f"{key}.json").write_text(json.dumps(data, indent=2, default=str))

    def load_from_cache(self, key: str) -> Optional[Any]:
        p = self.cache_dir / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def clear_cache(self) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink()

    # ---------- convenience ----------
    def get_uploaded_files(self) -> List[str]:
        return [p.name for p in self.upload_dir.iterdir() if p.is_file()]

    def get_output_files(self) -> List[str]:
        return [p.name for p in self.output_dir.iterdir() if p.is_file()]


class Logger:
    """Simple structured logger that writes to both console and rotating daily file."""
    def __init__(self, name: str = "pipeline", log_dir: str = "data/logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # avoid duplicate handlers on hot‑reload
        if self.logger.handlers:
            self.logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)

        fh = logging.FileHandler(log_dir / f"{name}_{datetime.now():%Y%m%d}.log")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

    # passthrough helpers
    def info(self, msg: str, **kw):     self.logger.info(msg, extra=kw)
    def warning(self, msg: str, **kw):  self.logger.warning(msg, extra=kw)
    def error(self, msg: str, **kw):    self.logger.error(msg, extra=kw)
    def debug(self, msg: str, **kw):    self.logger.debug(msg, extra=kw)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

class TextProcessor:
    """Light‑weight text cleaning, chunking, keyword & readability utilities."""
    @staticmethod
    def clean_text(text: str) -> str:
        import re
        text = " ".join(text.split())                             # collapse whitespace
        return re.sub(r"[^\w\s.,;:!?()\[\]{}\"'\\/\\-]", "", text).strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if end < len(text):  # try to cut on sentence boundary
                last = chunk.rfind(".")
                if last > chunk_size * 0.8:
                    chunk = chunk[: last + 1]
                    end = start + len(chunk)

            chunks.append(chunk)
            start = end - overlap
        return chunks

    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        import re
        from collections import Counter

        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        stop = {
            "that", "with", "have", "this", "will", "been", "from", "they", "know", "want",
            "good", "much", "some", "time", "very", "when", "come", "here", "just", "like",
            "long", "make", "many", "over", "such", "take", "than", "them", "well", "were",
            "your", "more", "also", "back", "other", "into", "after", "first", "never",
            "these", "think", "where", "being", "every", "great", "might", "shall", "still",
            "those", "under", "while",
        }
        counts = Counter(w for w in words if w not in stop)
        return [w for w, _ in counts.most_common(top_k)]

    @staticmethod
    def calculate_readability(text: str) -> Dict[str, float]:
        import re
        sentences = max(1, len(re.findall(r"[.!?]+", text)))
        words = text.split()
        syl = 0
        for w in words:
            w = w.lower().strip(".!,?")
            vowels, cnt, prev = "aeiouy", 0, False
            for c in w:
                if c in vowels and not prev:
                    cnt += 1
                    prev = True
                else:
                    prev = False
            if w.endswith("e") and cnt > 1:
                cnt -= 1
            syl += max(cnt, 1)

        total_words = max(1, len(words))
        flesch = 206.835 - 1.015 * (total_words / sentences) - 84.6 * (syl / total_words)
        fk_grade = 0.39 * (total_words / sentences) + 11.8 * (syl / total_words) - 15.59
        return {
            "flesch_reading_ease": round(max(0, min(100, flesch)), 2),
            "flesch_kincaid_grade": round(max(0, fk_grade), 2),
            "words": total_words,
            "sentences": sentences,
            "syllables": syl,
        }

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

class DataExporter:
    """Write data to CSV / Excel / JSON / Markdown."""
    @staticmethod
    def to_csv(data: List[Dict[str, Any]], filename: str) -> str:
        pd.DataFrame(data).to_csv(filename, index=False)
        return filename

    @staticmethod
    def to_excel(data: Dict[str, List[Dict[str, Any]]], filename: str) -> str:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for sheet, rows in data.items():
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)
        return filename

    @staticmethod
    def to_json(data: Any, filename: str) -> str:
        Path(filename).write_text(json.dumps(data, indent=2, default=str))
        return filename

    @staticmethod
    def to_markdown(data: Dict[str, Any], filename: str) -> str:
        lines = [
            "# Pipeline Results Report",
            f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
        ]
        for section, content in data.items():
            lines.append(f"## {section.replace('_', ' ').title()}")
            if isinstance(content, dict):
                for k, v in content.items():
                    lines.append(f"**{k}:** {v}")
            elif isinstance(content, list):
                lines.extend(f"- {item}" for item in content)
            else:
                lines.append(str(content))
            lines.append("")
        Path(filename).write_text("\n".join(lines))
        return filename

# ---------------------------------------------------------------------------
# Backwards‑compatibility shims (fixes the import error)
# ---------------------------------------------------------------------------

# Old names that the rest of the codebase expects
ResultsExporter = DataExporter
CacheManager = FileManager

# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def validate_api_keys() -> Dict[str, bool]:
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "huggingface": bool(os.getenv("HUGGINGFACE_API_KEY")),
    }


def estimate_tokens(text: str) -> int:
    return len(text) // 4  # ≈1 token / 4 chars


def calculate_cost(tokens: int, model: str = "gpt-3.5-turbo") -> float:
    pricing = {
        "gpt-3.5-turbo": 0.0015 / 1000,
        "gpt-4": 0.03 / 1000,
        "gpt-4-turbo": 0.01 / 1000,
        "claude-3-haiku": 0.00025 / 1000,
        "claude-3-sonnet": 0.003 / 1000,
        "claude-3-opus": 0.015 / 1000,
    }
    return tokens * pricing.get(model, 0.002 / 1000)


def format_results_for_display(results: Dict[str, Any]) -> Dict[str, Any]:
    formatted: Dict[str, Any] = {}
    for k, v in results.items():
        if isinstance(v, dict):
            formatted[k] = format_results_for_display(v)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            formatted[k] = pd.DataFrame(v)
        else:
            formatted[k] = v
    return formatted


# Expose public names
__all__ = [
    "FileManager",
    "CacheManager",
    "TextProcessor",
    "Logger",
    "DataExporter",
    "ResultsExporter",
    "validate_api_keys",
    "estimate_tokens",
    "calculate_cost",
    "format_results_for_display",
]
