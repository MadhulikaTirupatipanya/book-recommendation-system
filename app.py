import os
import re
from typing import Any

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Google Books Recommendation System",
    page_icon="📚",
    layout="wide",
)

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
DEFAULT_RESULTS = 40  # Google Books API maximum is 40 per request.


# ============================================================
# Google Books API
# ============================================================
def get_api_key() -> str:
    """Read an optional API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GOOGLE_BOOKS_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GOOGLE_BOOKS_API_KEY", "")


@st.cache_data(ttl=3600, show_spinner=False)
def search_google_books(query: str, max_results: int = DEFAULT_RESULTS) -> pd.DataFrame:
    """Search public book metadata through the Google Books Volumes API."""
    query = query.strip()
    if not query:
        return pd.DataFrame()

    params: dict[str, Any] = {
        "q": query,
        "maxResults": max(1, min(int(max_results), 40)),
        "printType": "books",
        "orderBy": "relevance",
    }

    api_key = get_api_key()
    if api_key:
        params["key"] = api_key

       response = requests.get(GOOGLE_BOOKS_URL, params=params, timeout=20)

    if response.status_code != 200:
        try:
            error = response.json().get("error", {})
            details = error.get("errors", [])
            reason = details[0].get("reason", "unknown") if details else "unknown"
            message = error.get("message", "No detailed error message")

            raise RuntimeError(
                f"Google Books API: HTTP {response.status_code}; "
                f"reason={reason}; message={message}"
            )
        except ValueError:
            raise RuntimeError(
                f"Google Books API returned HTTP {response.status_code}; "
                "could not parse its error response."
            )

    response.raise_for_status()
    payload = response.json()
    rows = []
    for item in payload.get("items", []):
        info = item.get("volumeInfo", {}) or {}
        image_links = info.get("imageLinks", {}) or {}
        authors = info.get("authors", []) or []
        categories = info.get("categories", []) or []

        title = info.get("title", "Title not available")
        description = info.get("description", "") or ""
        authors_text = ", ".join(authors) if authors else "Author not available"
        categories_text = ", ".join(categories) if categories else "Category not available"

        # Include title, authors, categories, and description in the text used
        # by TF-IDF. This keeps the original content-based approach.
        clean_text = " ".join(
            [str(title), authors_text, categories_text, description]
        )
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        rows.append(
            {
                "book_id": item.get("id", ""),
                "book_title": title,
                "author": authors_text,
                "average_rating": info.get("averageRating"),
                "ratings_count": info.get("ratingsCount"),
                "genres": categories_text,
                "book_details": description,
                "clean_text": clean_text,
                "thumbnail": image_links.get("thumbnail")
                or image_links.get("smallThumbnail"),
                "preview_link": info.get("previewLink", ""),
                "info_link": info.get("infoLink", ""),
                "published_date": info.get("publishedDate", "Not available"),
                "publisher": info.get("publisher", "Not available"),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# TF-IDF + cosine similarity
# ============================================================
def recommend_books(
    data: pd.DataFrame, selected_index: int, top_n: int = 5
) -> pd.DataFrame:
    """Rank the other API search results by text similarity to the selected book."""
    if data.empty or len(data) < 2:
        return pd.DataFrame()

    texts = data["clean_text"].fillna("").astype(str)
    if not texts.str.strip().any():
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)

    scores = cosine_similarity(matrix[selected_index], matrix).flatten()
    ranked_indices = scores.argsort()[::-1]
    ranked_indices = [
        idx for idx in ranked_indices if idx != selected_index
    ][:top_n]

    recommendations = data.iloc[ranked_indices].copy()
    recommendations["Similarity Score"] = [
        round(float(scores[idx]), 3) for idx in ranked_indices
    ]
    return recommendations.reset_index(drop=True)


def show_book_card(book: pd.Series, show_similarity: bool = False) -> None:
    """Display a book's cover and available metadata."""
    cover_col, details_col = st.columns([1, 4])

    with cover_col:
        if isinstance(book.get("thumbnail"), str) and book["thumbnail"]:
            # Google Books may return HTTP image URLs; use HTTPS where possible.
            image_url = book["thumbnail"].replace("http://", "https://")
            st.image(image_url, use_container_width=True)
        else:
            st.caption("Cover unavailable")

    with details_col:
        st.markdown(f"### {book.get('book_title', 'Title not available')}")
        st.write(f"**Author:** {book.get('author', 'Author not available')}")
        st.write(f"**Category:** {book.get('genres', 'Category not available')}")

        rating = book.get("average_rating")
        rating_count = book.get("ratings_count")
        if pd.notna(rating):
            rating_text = f"⭐ {rating}"
            if pd.notna(rating_count):
                rating_text += f" ({int(rating_count):,} ratings)"
            st.write(f"**Google Books rating:** {rating_text}")
        else:
            st.write("**Google Books rating:** Not available")

        st.write(
            f"**Published:** {book.get('published_date', 'Not available')}  "
            f" | **Publisher:** {book.get('publisher', 'Not available')}"
        )

        if show_similarity:
            st.write(f"**Text similarity score:** {book.get('Similarity Score', 0)}")

        description = str(book.get("book_details") or "").strip()
        if description:
            st.write(description[:700] + ("..." if len(description) > 700 else ""))
        else:
            st.write("Description not available.")

        link = book.get("preview_link") or book.get("info_link")
        if isinstance(link, str) and link:
            st.link_button("View on Google Books", link)

    st.divider()


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("📖 About")
st.sidebar.info(
    """
    ### Google Books Recommendation System

    - Live book metadata from Google Books
    - Text Mining
    - TF-IDF Vectorization
    - Content-Based Filtering
    - Cosine Similarity
    """
)
st.sidebar.caption(
    "Built with Python, Streamlit, Pandas, Requests and Scikit-learn."
)

api_key_present = bool(get_api_key())
if not api_key_present:
    st.sidebar.warning(
        "For reliable API access, add a Google Books API key to "
        ".streamlit/secrets.toml. Instructions are below the code."
    )


# ============================================================
# Main UI
# ============================================================
st.title("📚 Intelligent Book Recommendation System")
st.write(
    "Search Google Books for a title, author, or topic, select a result, "
    "and discover similar books using TF-IDF and cosine similarity."
)

query = st.text_input(
    "🔎 Search Google Books",
    placeholder="e.g., The Silent Patient, Agatha Christie, machine learning",
)
max_results = st.slider(
    "Number of books to retrieve for ranking",
    min_value=10,
    max_value=40,
    value=40,
    step=10,
)

if st.button("Search Books", type="primary"):
    if not query.strip():
        st.warning("Enter a book title, author, or topic first.")
    else:
        try:
            with st.spinner("Searching Google Books..."):
                results = search_google_books(query, max_results)

            if results.empty:
                st.warning(
                    "No books were returned. Try a different title, author, "
                    "or broader search phrase."
                )
            else:
                # Keep the current results in session state so the user can
                # choose a title and request recommendations on the next rerun.
                st.session_state["google_books_results"] = results
                st.session_state["google_books_query"] = query.strip()
        except requests.exceptions.Timeout:
            st.error("Google Books took too long to respond. Please try again.")
        except requests.exceptions.HTTPError as exc:
            st.error(
                f"Google Books returned an HTTP error: {exc}. "
                "Check your API key/quota or try again later."
            )
        except requests.exceptions.RequestException as exc:
            st.error(
                "Could not connect to Google Books. Check your internet "
                f"connection and try again. Details: {exc}"
            )
        except (ValueError, KeyError) as exc:
            st.error(f"Could not read the Google Books response: {exc}")

results = st.session_state.get("google_books_results")

if isinstance(results, pd.DataFrame) and not results.empty:
    st.markdown("---")
    st.subheader(f"Search results for: {st.session_state.get('google_books_query', '')}")
    st.caption(
        f"{len(results)} books retrieved. Recommendations are ranked only "
        "within these search results, not the entire Google Books catalogue."
    )

    # Use unique labels even when editions share the same title.
    labels = []
    for i, row in results.iterrows():
        label = f"{row['book_title']} — {row['author']}"
        labels.append(f"{label} [{i + 1}]")

    selected_label = st.selectbox("📚 Choose a book to find similar titles", labels)
    selected_index = labels.index(selected_label)

    st.markdown("#### Selected book")
    show_book_card(results.iloc[selected_index])

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
        key="recommendation_count",
    )

    if st.button("✨ Recommend Similar Books", type="primary"):
        if len(results) < 2:
            st.warning(
                "Google Books returned only one result. Search for a broader "
                "topic to retrieve more candidates."
            )
        else:
            with st.spinner("Ranking similar books..."):
                recommendations = recommend_books(results, selected_index, top_n)

            if recommendations.empty:
                st.warning("No recommendations could be generated.")
            else:
                st.subheader(
                    f"Recommendations similar to "
                    f"'{results.iloc[selected_index]['book_title']}'"
                )
                for _, book in recommendations.iterrows():
                    show_book_card(book, show_similarity=True)

st.caption(
    "Book metadata and available covers/links are supplied by Google Books. "
    "Ratings, categories, descriptions, and covers may be missing for some books."
)
