import os
from .sentiment import score_text

MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")


def llm_enabled():
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def llm_status():
    if llm_enabled():
        return f"Summaries: Claude ({MODEL})"
    return "Summaries: offline extractive mode (set ANTHROPIC_API_KEY to enable Claude)"


def _extractive_summary(product_name, reviews):
    scored = sorted(reviews, key=score_text)
    negatives = scored[:2]
    positives = scored[-2:][::-1]
    lines = [f"Based on {len(reviews)} reviews of {product_name}:", "", "Most positive reviews:"]
    lines += [f"- {r}" for r in positives]
    lines += ["", "Most critical reviews:"]
    lines += [f"- {r}" for r in negatives]
    return "\n".join(lines)


def _claude_summary(product_name, reviews, request):
    import anthropic
    client = anthropic.Anthropic()
    prompt = (
        f'You are analysing customer reviews for the product "{product_name}".\n'
        "Using ONLY the reviews below, write a concise summary with:\n"
        "1. Overall sentiment (one sentence)\n"
        "2. Top 3 strengths\n"
        "3. Top 3 complaints\n"
        "4. Who this product suits best\n"
        "Do not invent facts that are not in the reviews.\n"
        + (f"The user specifically asked: {request}\n" if request else "")
        + "\nReviews:\n"
        + "\n".join(f"- {r}" for r in reviews)
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def summarize_reviews(product_name, reviews, request=""):
    reviews = [r for r in reviews if r][:60]
    if not reviews:
        return "No reviews available for this product."
    if llm_enabled():
        try:
            return _claude_summary(product_name, reviews, request)
        except Exception as exc:
            return _extractive_summary(product_name, reviews) + f"\n\n(Claude call failed: {exc})"
    return _extractive_summary(product_name, reviews)
