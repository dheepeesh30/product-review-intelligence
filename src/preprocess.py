import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
NOISE_RE = re.compile(r"[^A-Za-z0-9\s.,!?'\-]")
SPACE_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

REQUIRED_COLUMNS = {"product_id", "product_name", "review_text"}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NOISE_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def split_sentences(text):
    return [s for s in SENT_SPLIT_RE.split(text) if s.strip()]


def load_reviews(source):
    """source: a file path or a file-like object containing CSV data."""
    df = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["review_text"]).copy()
    df["review_text"] = df["review_text"].astype(str)
    df["clean_text"] = df["review_text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["product_id", "clean_text"])
    return df.reset_index(drop=True)
