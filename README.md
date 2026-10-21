# AI Product Review Intelligence (Project 8)

NLP + Generative AI system that analyses customer reviews:
- Cleans and preprocesses review text
- Sentiment classification (positive / neutral / negative) with VADER
- Aspect-based sentiment (battery, performance, price, quality, durability, design, support)
- Product-level statistics and comparison
- TF-IDF semantic search over reviews (embedding-style retrieval)
- Summaries via Claude when ANTHROPIC_API_KEY is set, otherwise an offline extractive summary
- Preference-based product recommendations
- Streamlit UI and an evaluation script

## Quick start

    python -m venv .venv
    # Windows:  .venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

Then open the URL shown in the terminal (usually http://localhost:8501).

## Optional: enable Claude summaries

    # macOS/Linux
    export ANTHROPIC_API_KEY="your-key"
    # Windows PowerShell
    $env:ANTHROPIC_API_KEY="your-key"

Optionally set LLM_MODEL to override the model (default: claude-haiku-5-5).

## Use your own data

Upload any CSV from the sidebar. Required columns: product_id, product_name, review_text.
Optional: rating (1-5), used for the evaluation script and star averages.

## Evaluate sentiment accuracy

    python -m src.evaluate                 # uses data/sample_reviews.csv
    python -m src.evaluate path/to/file.csv

Ratings are mapped to labels (4-5 positive, 3 neutral, 1-2 negative) and compared with VADER predictions.

## Project layout

    app.py                  Streamlit UI
    src/preprocess.py       Cleaning, loading, sentence splitting
    src/sentiment.py        VADER sentiment scoring
    src/aspects.py          Aspect keywords and aspect-level sentiment
    src/analysis.py         Review-level analysis and product summaries
    src/retrieval.py        TF-IDF semantic search index
    src/recommender.py      Preference-based ranking
    src/llm.py              Claude summaries with offline fallback
    src/evaluate.py         Accuracy and F1 evaluation
    data/sample_reviews.csv Sample dataset (4 products, 40 reviews)

## Notes

- The sample data is synthetic and for demonstration only.
- For stronger embeddings, swap TF-IDF in src/retrieval.py for sentence-transformers.
- Scores are indicative. Verify important decisions against full review data.
