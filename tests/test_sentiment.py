"""Unit tests for the sentiment scoring and aggregation functions."""

from unittest.mock import MagicMock, patch
import pytest

from pipeline.sentiment import (
    calculate_daily_aggregate,
    score_headline_finbert,
    score_headline_vader,
    score_headlines_finbert,
    score_headlines_vader,
)


@pytest.fixture
def mock_finbert_pipeline():
    """Fixture to mock the Hugging Face pipeline call."""
    mock_nlp = MagicMock()

    # Define a side_effect function to simulate FinBERT responses
    def side_effect(inputs):
        if isinstance(inputs, str):
            inputs_list = [inputs]
        else:
            inputs_list = inputs

        outputs = []
        for text in inputs_list:
            text_lower = text.lower()
            if "good" in text_lower or "great" in text_lower or "positive" in text_lower:
                outputs.append({"label": "positive", "score": 0.95})
            elif "bad" in text_lower or "terrible" in text_lower or "negative" in text_lower:
                outputs.append({"label": "negative", "score": 0.85})
            else:
                outputs.append({"label": "neutral", "score": 0.60})
        return outputs

    mock_nlp.side_effect = side_effect
    with patch("pipeline.sentiment.pipeline", return_value=mock_nlp) as mock_p:
        yield mock_p


@patch("pipeline.sentiment.get_finbert_pipeline")
def test_score_headlines_finbert_success(mock_get_pipeline) -> None:
    """Test FinBERT scoring returns correct formats and weighted metrics."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {"label": "positive", "score": 0.95},
        {"label": "negative", "score": 0.85},
        {"label": "neutral", "score": 0.90},
    ]
    mock_get_pipeline.return_value = mock_pipe

    headlines = [
        "Market rises on high demand",
        "Stock crashes after warning",
        "Nothing happens today",
    ]
    results = score_headlines_finbert(headlines)

    assert len(results) == 3
    # Check positive mapping
    assert results[0]["label"] == "positive"
    assert results[0]["score"] == 0.95
    assert results[0]["weighted"] == 0.95

    # Check negative mapping
    assert results[1]["label"] == "negative"
    assert results[1]["score"] == 0.85
    assert results[1]["weighted"] == -0.85

    # Check neutral mapping
    assert results[2]["label"] == "neutral"
    assert results[2]["score"] == 0.90
    assert results[2]["weighted"] == 0.0


def test_score_headlines_vader_success() -> None:
    """Test VADER sentiment scoring results and label classifications."""
    headlines = [
        "This is an absolutely amazing performance!",
        "Disastrous results and terrible outlook.",
        "The board held their scheduled monthly meeting.",
    ]

    results = score_headlines_vader(headlines)

    assert len(results) == 3
    # First headline is positive
    assert results[0]["label"] == "positive"
    assert results[0]["weighted"] > 0.05

    # Second headline is negative
    assert results[1]["label"] == "negative"
    assert results[1]["weighted"] < -0.05

    # Third headline is neutral
    assert results[2]["label"] == "neutral"
    assert -0.05 <= results[2]["weighted"] <= 0.05


def test_score_headline_finbert_positive(mock_finbert_pipeline):
    """Test that a positive headline is scored correctly by FinBERT mock."""
    headline = "This stock is looking very good and has great growth potential."
    result = score_headline_finbert(headline)
    assert result["label"] == "positive"
    assert pytest.approx(result["score"]) == 0.95
    assert pytest.approx(result["weighted"]) == 0.95


def test_score_headline_finbert_negative(mock_finbert_pipeline):
    """Test that a negative headline is scored correctly by FinBERT mock."""
    headline = "This stock is bad and terrible, heading to negative territory."
    result = score_headline_finbert(headline)
    assert result["label"] == "negative"
    assert pytest.approx(result["score"]) == 0.85
    assert pytest.approx(result["weighted"]) == -0.85


def test_score_headline_finbert_neutral(mock_finbert_pipeline):
    """Test that a neutral headline is scored correctly by FinBERT mock."""
    headline = "The company announced it will release earnings tomorrow."
    result = score_headline_finbert(headline)
    assert result["label"] == "neutral"
    assert pytest.approx(result["score"]) == 0.60
    assert pytest.approx(result["weighted"]) == 0.0


def test_score_headlines_finbert_batch_size(mock_finbert_pipeline):
    """Test batched scoring with FinBERT mock, verifying it handles multiple headlines."""
    headlines = [
        "Good growth",
        "Bad news",
        "Neutral announcement",
        "Very great indeed",
    ]
    results = score_headlines_finbert(headlines, batch_size=2)
    assert len(results) == 4
    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"
    assert results[2]["label"] == "neutral"
    assert results[3]["label"] == "positive"


def test_score_headlines_finbert_empty_and_whitespace(mock_finbert_pipeline):
    """Test handling of empty or whitespace headlines in FinBERT."""
    assert score_headline_finbert("") == {"label": "neutral", "score": 0.0, "weighted": 0.0}
    assert score_headline_finbert("   ") == {"label": "neutral", "score": 0.0, "weighted": 0.0}
    assert score_headlines_finbert([]) == []


def test_score_headline_vader_individual():
    """Test VADER individual scoring with known positive, negative, and neutral statements."""
    pos_headline = "Excellent quarterly results and fantastic growth for this outstanding firm!"
    neg_headline = "Worst failure, terrible loss, and absolutely horrible performance."
    neu_headline = "The corporate meeting was scheduled for 10:00 AM in the main building."

    pos_res = score_headline_vader(pos_headline)
    neg_res = score_headline_vader(neg_headline)
    neu_res = score_headline_vader(neu_headline)

    # Positive assertions
    assert pos_res["label"] == "positive"
    assert pos_res["score"] > 0.0

    # Negative assertions
    assert neg_res["label"] == "negative"
    assert neg_res["score"] > 0.0

    # Neutral assertions
    assert neu_res["label"] == "neutral"
    assert neu_res["score"] > 0.0


def test_score_headline_vader_empty():
    """Test VADER handles empty strings gracefully."""
    assert score_headline_vader("") == {"label": "neutral", "score": 0.0, "weighted": 0.0}
    assert score_headline_vader("   ") == {"label": "neutral", "score": 0.0, "weighted": 0.0}


def test_calculate_daily_aggregate():
    """Test calculate_daily_aggregate calculations under various scenarios.

    Formula: sum(score * direction) / total_count
    """
    # 1. Empty scores list
    assert calculate_daily_aggregate([]) == 0.0

    # 2. Balanced positive and negative sentiments with same weight
    scores = [
        {"label": "positive", "score": 0.8},
        {"label": "negative", "score": 0.8},
    ]
    # (0.8 * 1.0 + 0.8 * -1.0) / 2 = 0.0
    assert pytest.approx(calculate_daily_aggregate(scores)) == 0.0

    # 3. All positive sentiments
    scores = [
        {"label": "positive", "score": 0.9},
        {"label": "positive", "score": 0.7},
    ]
    # (0.9 * 1.0 + 0.7 * 1.0) / 2 = 0.8
    assert pytest.approx(calculate_daily_aggregate(scores)) == 0.8

    # 4. Mixed sentiments including neutral
    scores = [
        {"label": "positive", "score": 0.8},  # weighted: 0.8
        {"label": "negative", "score": 0.6},  # weighted: -0.6
        {"label": "neutral", "score": 0.9},   # weighted: 0.0
    ]
    # (0.8 * 1.0 + 0.6 * -1.0 + 0.9 * 0.0) / 3 = 0.2 / 3 = 0.0666666...
    assert pytest.approx(calculate_daily_aggregate(scores)) == 0.2 / 3.0

    # 5. All negative sentiments
    scores = [
        {"label": "negative", "score": 0.5},
        {"label": "negative", "score": 0.9},
    ]
    # (-0.5 - 0.9) / 2 = -0.7
    assert pytest.approx(calculate_daily_aggregate(scores)) == -0.7
