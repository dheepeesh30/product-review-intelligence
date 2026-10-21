from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ReviewIndex:
    """TF-IDF vectors for semantic-style retrieval over reviews.
    Swap the vectorizer for sentence-transformers embeddings for stronger matching."""

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.df["clean_text"])

    def search(self, query, k=5, product_id=None):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        results = []
        for i in sims.argsort()[::-1]:
            if sims[i] <= 0 or len(results) >= k:
                break
            row = self.df.iloc[i]
            if product_id is not None and row["product_id"] != product_id:
                continue
            results.append((float(sims[i]), row))
        return results
