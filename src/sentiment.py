from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text):
    """Compound score from -1 (very negative) to +1 (very positive)."""
    return _analyzer.polarity_scores(text)["compound"]


def label_from_score(score, pos=0.05, neg=-0.05):
    if score >= pos:
        return "positive"
    if score <= neg:
        return "negative"
    return "neutral"
