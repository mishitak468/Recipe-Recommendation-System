
from __future__ import annotations

from engine import RecipeRecommender
import os
import sys
import time
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="AI Kitchen Scout", layout="wide", page_icon="🍳")

# Sidebars
st.sidebar.header("🎯 Filters")
max_time = st.sidebar.slider("Max time (mins)", 10, 120, 45)
max_ing = st.sidebar.slider("Max ingredients", 3, 20, 12)
max_cal = st.sidebar.slider("Calorie limit", 100, 1500, 600)

st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ System Health")
if "last_metrics" in st.session_state:
    lat = st.session_state["last_metrics"].get(
        "timings_ms", {}).get("total_ms", 0)
    if lat < 200:
        st.sidebar.success(f"Ultra-low latency: {lat:.0f}ms")
    elif lat < 500:
        st.sidebar.warning(f"Standard latency: {lat:.0f}ms")
    else:
        st.sidebar.error(f"High load: {lat:.0f}ms")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Algorithm")
k = st.sidebar.selectbox("Results (K)", [3, 5, 10], index=1)
lambda_mmr = st.sidebar.slider(
    "Diversity vs. Relevance",
    0.0, 1.0, 0.6, step=0.05,
    help="1.0 = pure relevance, 0.0 = maximise variety (MMR)"
)
tfidf_weight = st.sidebar.slider(
    "Content vs. Ratings weight",
    0.0, 1.0, 0.7, step=0.05,
    help="1.0 = ingredients only, 0.0 = ratings only"
)

# Load engine


@st.cache_resource(show_spinner="Loading recipe database…")
def load_engine():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "..", "data", "RAW_recipes.csv")
    return RecipeRecommender(path, sample_size=15000)


engine = load_engine()
interaction_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)
                    ), "..", "data", "RAW_interactions.csv"
)

# Tabs
tab_search, tab_metrics, tab_eval = st.tabs([
    "🍳 Recommendations", "📊 Metrics Dashboard", "🧪 Offline Evaluation"
])

# TAB 1 — RECOMMENDATIONS
with tab_search:
    st.title("🍳 AI Kitchen Scout")
    st.caption(
        "Hybrid recommender · TF-IDF + Jaccard + Bayesian ratings + MMR diversity reranking")

    mode = st.radio("Search mode:", [
                    "Similar to a recipe", "What's in my pantry?"], horizontal=True)
    pantry_input = search_query = None

    if mode == "Similar to a recipe":
        search_query = st.selectbox(
            "I love this recipe:", engine.df["name"].tolist())
    else:
        pantry_input = st.text_input(
            "Ingredients I have (comma-separated):",
            placeholder="chicken, garlic, spinach, lemon"
        )

    if st.button("🔍 Generate recommendations", type="primary"):
        with st.spinner("Computing recommendations…"):
            t_wall = time.perf_counter()
            results, metrics = engine.recommend(
                title=search_query,
                interaction_path=interaction_path,
                max_mins=max_time,
                max_ing=max_ing,
                max_cals=max_cal,
                pantry_input=pantry_input,
                k=k,
                lambda_mmr=lambda_mmr,
                tfidf_weight=tfidf_weight,
            )
            wall_ms = (time.perf_counter() - t_wall) * 1000

        # Store metrics for the dashboard
        st.session_state["last_metrics"] = metrics
        st.session_state["last_results"] = results

        if results is None or (isinstance(results, pd.DataFrame) and results.empty):
            st.warning(
                "No recipes match these constraints. Try widening your filters.")
        else:
            # Inline metrics
            st.success(f"Found {len(results)} recipes in {wall_ms:.0f}ms")
            m_cols = st.columns(5)
            m_cols[0].metric(
                "Precision@K", f"{metrics.get('precision_at_k', 0):.0%}")
            m_cols[1].metric(
                "NDCG@K",      f"{metrics.get('ndcg_at_k', 0):.3f}")
            m_cols[2].metric("MRR",         f"{metrics.get('mrr', 0):.3f}")
            m_cols[3].metric(
                "Diversity",   f"{metrics.get('intra_list_diversity', 0):.3f}")
            m_cols[4].metric(
                "Total latency", f"{metrics.get('timings_ms', {}).get('total_ms', wall_ms):.0f}ms")

            st.markdown("---")

            # Result cards
            cols = st.columns(2)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    with st.expander(
                        f"{'🥇🥈🥉🏅🎖️'[min(i, 4)]} {row['name'].title()} "
                        f"({int(row['calories'])} kcal)"
                    ):
                        # Core stats
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rating",      f"{row['rating']:.1f}/5")
                        c2.metric("Time",        f"{int(row['minutes'])}m")
                        c3.metric("Ingredients", int(row["n_ingredients"]))

                        # Score breakdown — the FAANG signal
                        st.markdown("**Score breakdown**")
                        score_cols = st.columns(3)
                        score_cols[0].metric(
                            "Content score",
                            f"{row['content_score']:.3f}",
                            help="Blended TF-IDF + Jaccard ingredient similarity"
                        )
                        score_cols[1].metric(
                            "Smoothed rating",
                            f"{row['smoothed_rating']:.2f}",
                            help="Bayesian-adjusted rating (corrects cold-start bias)"
                        )
                        score_cols[2].metric(
                            "Hybrid score",
                            f"{row['hybrid_score']:.3f}",
                            help=f"Content×{tfidf_weight:.0%} + Rating×{1-tfidf_weight:.0%}"
                        )

                        # Cold-start warning
                        if row.get("review_count", 0) < 5:
                            st.warning(
                                f"⚠️ Only {int(row.get('review_count', 0))} reviews — "
                                "rating is Bayesian-smoothed toward the global mean."
                            )

                        # st.write(f"🧂 **Ingredients:** {row['ingredients']}")
                        st.markdown("**Ingredients**")
                        # Split the string and show as labels
                        ings = row['ingredients'].strip(
                            "[]").replace("'", "").split(", ")
                        # Show first 10 as code-tags
                        st.write(" ".join([f"`{i}`" for i in ings[:10]]))
