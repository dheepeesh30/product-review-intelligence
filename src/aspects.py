import re
from .preprocess import split_sentences
from .sentiment import score_text

ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "charging", "charged", "drain", "drains"],
    "performance": ["performance", "fast", "slow", "speed", "lag", "lags", "smooth", "powerful", "responsive", "crashing"],
    "price": ["price", "cost", "expensive", "cheap", "value", "worth", "affordable", "pricey"],
    "quality": ["quality", "build", "sturdy", "flimsy", "well made", "premium", "finish"],
    "durability": ["durable", "durability", "broke", "broken", "cracked", "crack", "scratches", "lasted", "burned"],
    "design": ["design", "look", "looks", "sleek", "color", "colour", "comfortable", "comfort", "fit", "display"],
    "support": ["support", "warranty", "service", "refund", "replaced", "helped"],
}

_PATTERNS = {
    aspect: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for aspect, kws in ASPECT_KEYWORDS.items()
}


def aspect_sentiment(text):
    """Average sentiment of the sentences that mention each aspect."""
    buckets = {}
    for sentence in split_sentences(text):
        for aspect, pattern in _PATTERNS.items():
            if pattern.search(sentence):
                buckets.setdefault(aspect, []).append(score_text(sentence))
    return {a: sum(v) / len(v) for a, v in buckets.items()}
