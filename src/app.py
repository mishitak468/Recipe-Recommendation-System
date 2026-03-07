from engine import RecipeRecommender
import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="AI Kitchen Scout", layout="wide", page_icon="🍳")

# --- SIDEBAR: ADVANCED FILTERS ---
st.sidebar.header("🎯 Target Your Meal")
max_time = st.sidebar.slider("Max Time (mins)", 10, 120, 45)
max_ing = st.sidebar.slider("Max Ingredients", 3, 20, 12)
max_cal = st.sidebar.slider("Calories Limit", 100, 1500, 600)

st.title("🍳 AI Kitchen Scout")
st.markdown("A Hybrid Recommender with Nutritional Filtering and Pantry Search.")


@st.cache_resource
def load_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    recipe_path = os.path.join(base_dir, '..', 'data', 'RAW_recipes.csv')
    return RecipeRecommender(recipe_path, sample_size=15000)


try:
    engine = load_engine()
    interaction_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', 'data', 'RAW_interactions.csv')

    # --- UI MODE SELECTION ---
    mode = st.radio("Choose search style:", [
                    "Similar to a Recipe", "What's in my Pantry?"])

    pantry_input = None
    search_query = None

    if mode == "Similar to a Recipe":
        search_query = st.selectbox(
            "I love this recipe:", engine.df['name'].tolist())
    else:
        pantry_input = st.text_input(
            "Enter ingredients you have (e.g., chicken, onion, spinach):")

    if st.button("Generate Recommendations"):
        results = engine.recommend(search_query, interaction_path,
                                   max_mins=max_time, max_ing=max_ing,
                                   max_cals=max_cal, pantry_input=pantry_input)

        if results is None or len(results) == 0:
            st.warning(
                "No recipes match these specific health and time constraints. Try widening your search!")
        else:
            st.success("Matching Recipes Found:")
            cols = st.columns(2)
            for i, (idx, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    with st.expander(f"📖 {row['name'].title()} ({int(row['calories'])} cals)"):
                        st.write(
                            f"⭐ **Rating:** {row['rating']:.1f}/5 | ⏱️ **Time:** {row['minutes']}m")
                        st.write(f"🧂 **Ingredients:** {row['ingredients']}")

except Exception as e:
    st.error(f"App Error: {e}")
