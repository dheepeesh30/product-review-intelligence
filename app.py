from io import StringIO
from pathlib import Path

import streamlit as st

from src.preprocess import load_reviews
from src.analysis import analyze_reviews, product_summary
from src.retrieval import ReviewIndex
from src.recommender import recommend
from src.llm import summarize_reviews
from src.aspects import ASPECT_KEYWORDS

SAMPLE = Path(__file__).parent / "data" / "sample_reviews.csv"

st.set_page_config(page_title="Product Review Intelligence", layout="wide")


@st.cache_data(show_spinner="Analysing reviews...")
def load_analysed(csv_text):
    return analyze_reviews(load_reviews(StringIO(csv_text)))


@st.cache_resource
def build_index(csv_text):
    return ReviewIndex(load_analysed(csv_text))


st.title("AI Product Review Intelligence")

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload a reviews CSV (optional)", type="csv")
csv_text = uploaded.getvalue().decode("utf-8") if uploaded else SAMPLE.read_text(encoding="utf-8")
st.sidebar.caption("Required columns: product_id, product_name, review_text. Optional: rating.")
st.sidebar.info(llm_status())

df = load_analysed(csv_text)
summary = product_summary(df)
index = build_index(csv_text)
names = dict(zip(summary["product_id"], summary["product_name"]))

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Compare", "Search & Summarise", "Recommend"])

with tab1:
    st.subheader("Product summary")
    st.dataframe(summary)
    st.subheader("Sentiment distribution (all reviews)")
    st.bar_chart(df["sentiment"].value_counts())

with tab2:
    chosen = st.multiselect(
        "Choose products to compare",
        options=list(names),
        default=list(names)[:2],
        format_func=lambda pid: names[pid],
    )
    if not chosen:
        st.info("Select at least one product.")
    else:
        sub = summary[summary["product_id"].isin(chosen)].set_index("product_name")
        aspect_scores = (
            sub[[f"{a}_score" for a in ASPECT_KEYWORDS]]
            .rename(columns=lambda c: c.replace("_score", ""))
            .T
        )
        st.subheader("Aspect sentiment (-1 to +1, blank = not mentioned)")
        st.dataframe(aspect_scores.round(3))
        st.bar_chart(aspect_scores.fillna(0))
        st.subheader("Overall")
        st.dataframe(sub[["reviews", "avg_sentiment", "pct_positive", "pct_negative"]])

with tab3:
    pid = st.selectbox("Product", options=list(names), format_func=lambda p: names[p])
    question = st.text_input("Ask about this product (optional)", placeholder="e.g. how is the battery life?")
    if question:
        hits = index.search(question, k=5, product_id=pid)
        if hits:
            st.subheader("Matching reviews")
            for score, row in hits:
                st.markdown(f"**{row['sentiment'].title()}** (match {score:.2f}): {row['review_text']}")
        else:
            st.info("No matching reviews found for that question.")
    if st.button("Summarise reviews"):
        texts = df[df["product_id"] == pid]["clean_text"].tolist()
        with st.spinner("Summarising..."):
            st.markdown(summarize_reviews(names[pid], texts, question))

with tab4:
    st.write("How much does each factor matter to you? (0 = ignore)")
    cols = st.columns(len(ASPECT_KEYWORDS))
    weights = {a: col.slider(a.title(), 0.0, 1.0, 0.0, 0.1) for col, a in zip(cols, ASPECT_KEYWORDS)}
    weights = {a: w for a, w in weights.items() if w > 0}
    if weights:
        st.subheader("Best matches for your priorities")
        st.dataframe(recommend(summary, weights))
    else:
        st.info("Move at least one slider above zero.")
