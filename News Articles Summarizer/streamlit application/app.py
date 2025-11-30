import streamlit as st
from newspaper import Article
import re
import math
from typing import List

st.set_page_config(page_title="News Article Details & Summary", layout="centered")
st.title("News Article Details & Summary")
st.write("Enter a URL to a news/article page and get the Title, Authors, Publication Date and a short Summary.")

# --- Detect optional heavy deps
def _transformers_available() -> bool:
    try:
        import importlib
        util = getattr(importlib, "util", None)
        if util and getattr(util, "find_spec", None):
            return util.find_spec("transformers") is not None
        try:
            import transformers  # type: ignore
            return True
        except Exception:
            return False
    except Exception:
        return False

_TRANSFORMERS_AVAILABLE = _transformers_available()

# --- UI controls
with st.form("article_form"):
    url = st.text_input("Article URL", placeholder="https://example.com/article")
    method_options = ["newspaper", "textrank", "first_n"]
    if _TRANSFORMERS_AVAILABLE:
        method_options.append("abstractive")
    method = st.selectbox("Summary method", method_options, index=1)
    max_sentences = st.slider("Max summary sentences", min_value=1, max_value=6, value=3)
    submitted = st.form_submit_button("Fetch & Summarize")


# --- Utilities
def _format_publish_date(dt):
    if not dt:
        return "N/A"
    if isinstance(dt, (str,)):
        return dt
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt)


def _split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


_STOPWORDS = {
    "the","and","is","in","to","of","a","for","on","that","with","as","are","it","by","an","be","this","from","at","or","which","was","has","have","but","not","were","they","their","its","will","can"
}


def _tokenize(sentence: str) -> List[str]:
    words = re.findall(r"\w+", sentence.lower())
    return [w for w in words if w not in _STOPWORDS]


def _sentence_similarity(s1: List[str], s2: List[str]) -> float:
    if not s1 or not s2:
        return 0.0
    set1, set2 = set(s1), set(s2)
    common = set1.intersection(set2)
    if not common:
        return 0.0
    denom = math.log(len(set1) + 1) + math.log(len(set2) + 1)
    if denom <= 0:
        return float(len(common))
    return float(len(common)) / denom


def _pagerank(sim_matrix: List[List[float]], eps: float = 1e-4, d: float = 0.85, max_iter: int = 100) -> List[float]:
    n = len(sim_matrix)
    if n == 0:
        return []
    scores = [1.0] * n
    for _ in range(max_iter):
        prev = scores.copy()
        for i in range(n):
            summation = 0.0
            for j in range(n):
                if i == j:
                    continue
                denom = sum(sim_matrix[j])
                if denom == 0:
                    continue
                summation += (sim_matrix[j][i] * prev[j]) / denom
            scores[i] = (1 - d) + d * summation
        diff = sum(abs(scores[i] - prev[i]) for i in range(n))
        if diff < eps:
            break
    return scores


def textrank_summary(text: str, n_sentences: int = 3) -> str:
    sentences = _split_into_sentences(text)
    if not sentences:
        return "(no summary available)"
    if len(sentences) <= n_sentences:
        return "\n\n".join(sentences)

    tokenized = [_tokenize(s) for s in sentences]
    n = len(sentences)
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _sentence_similarity(tokenized[i], tokenized[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim
    scores = _pagerank(sim_matrix)
    ranked_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)[:n_sentences]
    ranked_idx_set = set(ranked_idx)
    selected = [sentences[i] for i in range(n) if i in ranked_idx_set]
    return "\n\n".join(selected)


def first_n_summary(text: str, n_sentences: int = 3) -> str:
    sentences = _split_into_sentences(text)
    if not sentences:
        return "(no summary available)"
    return "\n\n".join(sentences[:n_sentences])


def try_load_abstractive_model():
    try:
        from transformers import pipeline
    except Exception:
        return None
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        return None


def abstractive_summary(transformer_pipe, text: str, n_sentences: int = 3) -> str:
    if transformer_pipe is None:
        return "(abstractive model not available)"
    max_length = min(512, max(60, n_sentences * 60))
    try:
        outputs = transformer_pipe(text, max_length=max_length, min_length=30, do_sample=False)
        if outputs and isinstance(outputs, list):
            return outputs[0].get("summary_text", "(no summary produced)")
    except Exception as e:
        return f"(abstractive summarization failed: {e})"
    return "(no summary produced)"


def summarize_article(text: str, method: str = "textrank", n_sentences: int = 3, transformer_pipe=None) -> str:
    if not text or not text.strip():
        return "(no summary available)"
    method = (method or "textrank").lower()
    if method == "newspaper":
        return first_n_summary(text, n_sentences)
    if method == "textrank":
        return textrank_summary(text, n_sentences)
    if method == "first_n":
        return first_n_summary(text, n_sentences)
    if method == "abstractive":
        return abstractive_summary(transformer_pipe, text, n_sentences)
    return first_n_summary(text, n_sentences)


# --- Main interaction
if submitted:
    if not url or not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Downloading and processing article..."):
            try:
                article = Article(url)
                article.download()
                article.parse()

                try:
                    article.nlp()
                except Exception:
                    pass

                title = article.title or "N/A"
                authors = ", ".join(article.authors) if article.authors else "N/A"
                publish_date = _format_publish_date(article.publish_date)

                raw_text = getattr(article, "text", "")
                summary = None
                if method == "newspaper":
                    summary = getattr(article, "summary", None)
                    if not summary:
                        summary = first_n_summary(raw_text, max_sentences)
                elif method == "abstractive" and _TRANSFORMERS_AVAILABLE:
                    transformer = try_load_abstractive_model()
                    if transformer is None:
                        st.warning("Abstractive model couldn't be loaded; falling back to extractive method.")
                        summary = textrank_summary(raw_text, max_sentences)
                    else:
                        summary = abstractive_summary(transformer, raw_text, max_sentences)
                else:
                    summary = summarize_article(raw_text, method, max_sentences)

                st.subheader("Title")
                st.write(title)

                st.subheader("Authors")
                st.write(authors)

                st.subheader("Publication Date")
                st.write(publish_date)

                st.subheader("Summary")
                st.write(summary)

            except Exception as e:
                st.error(f"Failed to process the article: {e}")
                st.exception(e)


# Helpful tips / footer
st.markdown("---")
st.markdown("**Tips:** If the extraction fails for some sites, try a different article URL or copy the article text into a local file and run a different processor.")
