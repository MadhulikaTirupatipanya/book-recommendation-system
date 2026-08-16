import ast
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Intelligent Book Recommendation System",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "books_demo.csv"


# ============================================================
# Load Dataset
# ============================================================

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)

    required_columns = [
        "book_title",
        "author",
        "average_rating",
        "genres",
        "book_details",
        "clean_text",
    ]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    return data


df = load_data(str(DATA_PATH))


# ============================================================
# Build TF-IDF Model
# ============================================================

@st.cache_resource
def build_model(data: pd.DataFrame):
    tfidf = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
    )

    tfidf_matrix = tfidf.fit_transform(data["clean_text"].fillna(""))

    indices = pd.Series(
        data.index,
        index=data["book_title"],
    ).drop_duplicates()

    return tfidf_matrix, indices


tfidf_matrix, indices = build_model(df)


# ============================================================
# Recommendation Function
# ============================================================

def clean_genres(value):
    if pd.isna(value):
        return "Not available"

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return ", ".join(str(item) for item in parsed)
        except (ValueError, SyntaxError):
            pass

    return str(value)


def recommend_books(title: str, top_n: int = 10):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix,
    ).flatten()

    # Exclude the selected book itself.
    ranked_indices = similarity_scores.argsort()[::-1]
    ranked_indices = [
        i for i in ranked_indices if i != idx
    ][:top_n]

    recommendations = df.iloc[ranked_indices][
        [
            "book_title",
            "author",
            "average_rating",
            "genres",
            "book_details",
        ]
    ].copy()

    recommendations["genres"] = recommendations["genres"].apply(clean_genres)

    recommendations["book_details"] = (
        recommendations["book_details"]
        .fillna("Description not available.")
        .astype(str)
        .apply(lambda x: x[:300] + "..." if len(x) > 300 else x)
    )

    recommendations["Similarity Score"] = [
        round(float(similarity_scores[i]), 3)
        for i in ranked_indices
    ]

    return recommendations.reset_index(drop=True)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📖 About")

st.sidebar.info(
    """
### Intelligent Book Recommendation System

This application uses:

- Text Mining
- TF-IDF Vectorization
- Content-Based Filtering
- Cosine Similarity
- Machine Learning
"""
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with Python, Streamlit, Pandas and Scikit-learn."
)


# ============================================================
# Main UI
# ============================================================

st.title("📚 Intelligent Book Recommendation System")

st.write(
    "Select a book and discover similar books based on "
    "the textual similarity of their descriptions and genres."
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Books in Demo Dataset", f"{len(df):,}")

with col2:
    st.metric("Recommendation Method", "TF-IDF + Cosine Similarity")

st.markdown("---")

selected_book = st.selectbox(
    "📚 Select a Book",
    sorted(df["book_title"].dropna().unique()),
)

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5,
)

if st.button("🔍 Recommend Books", type="primary"):
    with st.spinner("Finding similar books..."):
        recommendations = recommend_books(
            selected_book,
            top_n=top_n,
        )

    if recommendations.empty:
        st.error("No recommendations could be generated.")
    else:
        st.subheader(f"Top Recommendations for '{selected_book}'")

        for _, book in recommendations.iterrows():
            st.markdown(f"### 📖 {book['book_title']}")

            info_col1, info_col2 = st.columns([1, 2])

            with info_col1:
                st.write(f"**Author:** {book['author']}")
                st.write(f"**Genre:** {book['genres']}")
                st.write(f"**Average Rating:** ⭐ {book['average_rating']}")
                st.write(
                    f"**Similarity Score:** {book['Similarity Score']}"
                )

            with info_col2:
                st.write("**Description:**")
                st.write(book["book_details"])

            st.divider()


st.caption(
    "Developed using Python, Streamlit, Pandas, Scikit-learn, "
    "TF-IDF and Cosine Similarity."
)
