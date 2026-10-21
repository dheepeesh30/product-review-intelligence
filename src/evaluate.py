import sys
from sklearn.metrics import accuracy_score, classification_report, f1_score
from src.preprocess import load_reviews
from src.sentiment import score_text, label_from_score


def rating_to_label(r):
    if r >= 4:
        return "positive"
    if r <= 2:
        return "negative"
    return "neutral"


def main(path="data/sample_reviews.csv"):
    df = load_reviews(path)
    if "rating" not in df.columns or df["rating"].isna().all():
        sys.exit("Evaluation needs a numeric 'rating' column.")
    df = df.dropna(subset=["rating"])
    y_true = df["rating"].map(rating_to_label)
    y_pred = df["clean_text"].map(lambda t: label_from_score(score_text(t)))
    print(f"Reviews evaluated: {len(df)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    print()
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_reviews.csv")
