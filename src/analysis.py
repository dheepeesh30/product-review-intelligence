import pandas as pd
from .sentiment import score_text, label_from_score
from .aspects import aspect_sentiment, ASPECT_KEYWORDS


def analyze_reviews(df):
    df = df.copy()
    df["sentiment_score"] = df["clean_text"].map(score_text)
    df["sentiment"] = df["sentiment_score"].map(label_from_score)
    aspects = df["clean_text"].map(aspect_sentiment)
    for aspect in ASPECT_KEYWORDS:
        df[f"aspect_{aspect}"] = aspects.map(lambda d, a=aspect: d.get(a))
    return df


def product_summary(df):
    rows = []
    for (pid, name), grp in df.groupby(["product_id", "product_name"]):
        row = {
            "product_id": pid,
            "product_name": name,
            "reviews": len(grp),
            "avg_sentiment": round(grp["sentiment_score"].mean(), 3),
            "pct_positive": round((grp["sentiment"] == "positive").mean() * 100, 1),
            "pct_negative": round((grp["sentiment"] == "negative").mean() * 100, 1),
        }
        if "rating" in grp.columns and grp["rating"].notna().any():
            row["avg_rating"] = round(grp["rating"].mean(), 2)
        for aspect in ASPECT_KEYWORDS:
            vals = grp[f"aspect_{aspect}"].dropna()
            row[f"{aspect}_score"] = round(vals.mean(), 3) if len(vals) else None
            row[f"{aspect}_mentions"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)
