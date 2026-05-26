"""Sentiment classification engine using FinBERT and VADER models.

This module provides functions to score individual headlines using either FinBERT
(primary) or VADER (secondary comparative) models, along with weighted daily
aggregation logic.
"""

import logging
from typing import Dict, List, Union

import numpy as np
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set up module logger
logger = logging.getLogger(__name__)

# Singletons for models to avoid reloading them on each call
_finbert_pipeline = None
_vader_analyzer = None


def get_finbert_pipeline():
    """Lazy-load and return the FinBERT Hugging Face pipeline."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        logger.info("Initializing primary model ProsusAI/finbert pipeline...")
        try:
            # We explicitly specify text-classification task and the ProsusAI/finbert model.
            _finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT pipeline: {e}", exc_info=True)
            raise e
    return _finbert_pipeline


def _get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Lazy-load and return the VADER SentimentIntensityAnalyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        logger.info("Initializing VADER SentimentIntensityAnalyzer...")
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def score_headlines_finbert(
    headlines: List[str], batch_size: int = 16
) -> List[Dict[str, Union[str, float]]]:
    """Score a list of headlines using FinBERT in batches.

    Args:
        headlines (List[str]): A list of headline strings to score.
        batch_size (int): Batch size for batching headlines. Defaults to 16.

    Returns:
        List[Dict[str, Union[str, float]]]: A list of dictionaries containing keys
            'label' ('positive', 'negative', 'neutral'), 'score' (float), and
            'weighted' (float).
    """
    if not headlines:
        return []

    results: List[Dict[str, Union[str, float]]] = []
    try:
        nlp = get_finbert_pipeline()
    except Exception as e:
        logger.error(f"Skipping FinBERT batch scoring due to initialization failure: {e}")
        return [{"label": "neutral", "score": 0.0, "weighted": 0.0} for _ in headlines]

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        # Clean whitespaces
        cleaned_batch = [h.strip() if h else "" for h in batch]
        try:
            batch_results = nlp(cleaned_batch)
            for res in batch_results:
                label = str(res["label"]).lower()
                score = float(res["score"])
                # Calculate weighted score for easy access/compatibility
                if label == "positive":
                    weighted = score
                elif label == "negative":
                    weighted = -score
                else:
                    weighted = 0.0

                results.append({"label": label, "score": score, "weighted": weighted})
        except Exception as e:
            logger.error(f"Error scoring batch {i // batch_size + 1}: {e}", exc_info=True)
            # Default to neutral on batch failure
            for _ in batch:
                results.append({"label": "neutral", "score": 0.0, "weighted": 0.0})

    return results


def score_headline_finbert(headline: str) -> Dict[str, Union[str, float]]:
    """Score a single headline using FinBERT.

    Args:
        headline (str): The headline text to analyze.

    Returns:
        Dict[str, Union[str, float]]: A dictionary with 'label', 'score', and 'weighted'.
    """
    if not headline or not headline.strip():
        return {"label": "neutral", "score": 0.0, "weighted": 0.0}

    results = score_headlines_finbert([headline])
    if results:
        return results[0]
    return {"label": "neutral", "score": 0.0, "weighted": 0.0}


def score_headline_vader(headline: str) -> Dict[str, Union[str, float]]:
    """Score a single headline using VADER Sentiment Intensity Analyzer.

    VADER compound scores are mapped to positive, negative, and neutral labels
    using standard thresholds (positive: >= 0.05, negative: <= -0.05, neutral: otherwise).

    Args:
        headline (str): The headline text to analyze.

    Returns:
        Dict[str, Union[str, float]]: A dictionary with 'label', 'score', and 'weighted'.
    """
    if not headline or not headline.strip():
        return {"label": "neutral", "score": 0.0, "weighted": 0.0}

    try:
        analyzer = _get_vader_analyzer()
        scores = analyzer.polarity_scores(headline)
        compound = float(scores.get("compound", 0.0))

        if compound >= 0.05:
            return {"label": "positive", "score": compound, "weighted": compound}
        elif compound <= -0.05:
            return {"label": "negative", "score": abs(compound), "weighted": compound}
        else:
            # Return neutral. Use (1 - absolute compound) as the confidence/score.
            return {"label": "neutral", "score": 1.0 - abs(compound), "weighted": 0.0}
    except Exception as e:
        logger.error(f"Error scoring headline with VADER: {e}", exc_info=True)
        return {"label": "neutral", "score": 0.0, "weighted": 0.0}


def score_headlines_vader(headlines: List[str]) -> List[Dict[str, Union[str, float]]]:
    """Score a list of headlines using VADER sentiment analyzer.

    Args:
        headlines (List[str]): A list of headline strings to score.

    Returns:
        List[Dict[str, Union[str, float]]]: A list of dictionaries containing keys
            'label', 'score', and 'weighted'.
    """
    return [score_headline_vader(h) for h in headlines]


def calculate_daily_aggregate(scores: List[Dict[str, Union[str, float]]]) -> float:
    """Calculate the weighted average of confidence scores for a given day and ticker.

    Weighted Average = sum(confidence * direction) / Total Count
    where direction is +1.0 for positive, -1.0 for negative, and 0.0 for neutral.

    Args:
        scores (List[Dict[str, Union[str, float]]]): List of dictionaries with
            'label' and 'score' keys.

    Returns:
        float: A weighted average score in the range [-1.0, +1.0].
            Returns 0.0 if scores is empty.
    """
    if not scores:
        return 0.0

    total_weighted_score = 0.0
    for s in scores:
        label = str(s.get("label", "neutral")).lower()
        score = float(s.get("score", 0.0))

        if label == "positive":
            direction = 1.0
        elif label == "negative":
            direction = -1.0
        else:
            direction = 0.0

        total_weighted_score += score * direction

    weighted_avg = total_weighted_score / len(scores)
    # Clamp results to [-1.0, 1.0] just in case of precision issues
    return float(np.clip(weighted_avg, -1.0, 1.0))
