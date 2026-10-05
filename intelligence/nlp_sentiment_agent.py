"""
strategy/sentiment.py — NLP-based news sentiment analyzer

Uses a HuggingFace transformer model to classify financial news headlines
as positive (bullish USD) or negative (bearish USD), returning a composite
score between -1.0 and +1.0.

Model: distilbert-base-uncased-finetuned-sst-2-english (default)
  - Small (~260 MB), fast on CPU
  - Lazy-loaded on first call to avoid startup delay
  - Graceful degradation: returns 0.0 (neutral) if model fails to load
"""
from __future__ import annotations

import time
from typing import List, Optional

try:
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from core import config
from core.logger import get_logger

logger = get_logger("Sentiment")


class SentimentAnalyzer:
    """
    Analyzes financial news headlines using a pretrained NLP model.
    Returns a composite sentiment score between -1.0 (very bearish) and +1.0 (very bullish).
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_loaded = False
        self._model_failed = False
        self._cache: dict[str, float] = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = 300.0  # 5 minutes

    def _load_model(self) -> bool:
        """Lazy-load the sentiment analysis pipeline."""
        if self._model_loaded:
            return True
        if self._model_failed:
            return False

        try:
            logger.info("🧠 Loading sentiment model (first time only)...")

            if not _HAS_TRANSFORMERS:
                raise ImportError("transformers not installed")

            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=config.SENTIMENT_MODEL,
                device=-1,  # Force CPU
                top_k=None,  # Get all class scores
            )
            self._model_loaded = True
            logger.info("✅ Sentiment model loaded successfully")
            return True

        except ImportError:
            logger.warning(
                "transformers or torch not installed. "
                "Sentiment analysis disabled. "
                "Install: pip install transformers torch"
            )
            self._model_failed = True
            return False

        except Exception as exc:
            logger.error(f"Failed to load sentiment model: {exc}")
            self._model_failed = True
            return False

    def analyze(self, headlines: List[str]) -> float:
        """
        Analyze a list of news headlines and return a composite sentiment score.

        Args:
            headlines: List of news headline strings

        Returns:
            Score between -1.0 (very bearish/negative) and +1.0 (very bullish/positive).
            Returns 0.0 if no headlines or model unavailable.
        """
        if not headlines:
            return 0.0

        # Check cache
        cache_key = "|".join(sorted(headlines[:10]))
        now = time.time()
        if now - self._cache_time < self._cache_ttl and cache_key in self._cache:
            return self._cache[cache_key]

        # Load model if needed
        if not self._load_model():
            return 0.0

        try:
            scores = []
            for headline in headlines[:10]:  # Limit to 10 headlines
                score = self._analyze_single(headline)
                if score is not None:
                    scores.append(score)

            if not scores:
                return 0.0

            # Weighted average: more recent headlines (first in list) get higher weight
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            total_weight = sum(weights)
            composite = sum(s * w for s, w in zip(scores, weights)) / total_weight

            # Clamp to [-1, +1]
            composite = max(-1.0, min(1.0, composite))

            # Cache result
            self._cache[cache_key] = composite
            self._cache_time = now

            logger.info(
                f"🧠 Sentiment: {composite:+.3f} "
                f"({len(scores)} headlines analyzed)"
            )
            return composite

        except Exception as exc:
            logger.error(f"Sentiment analysis error: {exc}")
            return 0.0

    def _analyze_single(self, text: str) -> Optional[float]:
        """Analyze a single text and return score in [-1, +1]."""
        if not text.strip():
            return None

        try:
            # Truncate to model max length
            text = text[:512]

            results = self._pipeline(text)

            # results is a list of dicts: [{"label": "POSITIVE", "score": 0.99}, ...]
            # or a list of lists if top_k=None
            if isinstance(results[0], list):
                results = results[0]

            # Build score: POSITIVE → +, NEGATIVE → -
            pos_score = 0.0
            neg_score = 0.0
            for r in results:
                label = r["label"].upper()
                conf = r["score"]
                if label in ("POSITIVE", "POS", "LABEL_1"):
                    pos_score = conf
                elif label in ("NEGATIVE", "NEG", "LABEL_0"):
                    neg_score = conf

            # Map to [-1, +1]: positive bias → bullish USD
            # For financial context: positive news = bullish market = strong USD
            return pos_score - neg_score

        except Exception as exc:
            logger.debug(f"Error analyzing headline: {exc}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if the sentiment model is loaded and ready."""
        return self._model_loaded and not self._model_failed
