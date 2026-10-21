import pandas as pd


def recommend(summary, preferences, top_n=4):
    """preferences: {aspect: weight 0..1}. Ranks products by weighted aspect sentiment.
    Falls back to overall sentiment if a product has no mentions of the chosen aspects."""
    rows = []
    for _, r in summary.iterrows():
        num = den = 0.0
        for aspect, weight in preferences.items():
            s = r.get(f"{aspect}_score")
            if s is None or pd.isna(s):
                continue
            num += weight * s
            den += weight
        fit = num / den if den else r["avg_sentiment"]
        rows.append({
            "product_name": r["product_name"],
            "overall_sentiment": r["avg_sentiment"],
            "preference_fit": round(fit, 3),
        })
    out = pd.DataFrame(rows).sort_values("preference_fit", ascending=False)
    return out.head(top_n).reset_index(drop=True)
