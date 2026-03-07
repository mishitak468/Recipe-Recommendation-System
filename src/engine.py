import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecipeRecommender:
    def __init__(self, recipe_path, sample_size=15000):
        self.df = pd.read_csv(recipe_path).head(sample_size)
        # Parse ingredients and nutrition strings into usable lists/values
        self.df['clean_ingredients'] = self.df['ingredients'].apply(
            self._clean_ingredients)
        self.df['calories'] = self.df['nutrition'].apply(
            lambda x: ast.literal_eval(x)[0])

        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.df['clean_ingredients'])
        self.cosine_sim = cosine_similarity(
            self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(
            self.df.index, index=self.df['name']).drop_duplicates()

    def _clean_ingredients(self, x):
        ingredients = ast.literal_eval(x)
        return " ".join([i.replace(" ", "") for i in ingredients])

    def search_by_pantry(self, ingredients_input):
        # Convert user input like "chicken, garlic" into "chicken garlic"
        query = ingredients_input.replace(",", " ").lower()
        query_vec = self.tfidf.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return sim_scores

    def recommend(self, title, interaction_path, max_mins=60, max_ing=10, max_cals=500, pantry_input=None):
        # 1. Get Base Similarity Scores
        if pantry_input:
            sim_scores = self.search_by_pantry(pantry_input)
        elif title in self.indices:
            idx = self.indices[title]
            sim_scores = self.cosine_sim[idx]
        else:
            return None

        # 2. Hybrid Scoring with Ratings
        interactions = pd.read_csv(interaction_path)
        avg_ratings = interactions.groupby(
            'recipe_id')['rating'].mean().reset_index()
        candidates = self.df.merge(
            avg_ratings, left_on='id', right_on='recipe_id', how='left').fillna(0)

        candidates['similarity'] = sim_scores
        # 70% Ingredient Match + 30% User Rating
        candidates['hybrid_score'] = (
            candidates['similarity'] * 0.7) + ((candidates['rating'] / 5) * 0.3)

        # 3. Multi-Objective Filtering (Time, Ingredients, Calories)
        filtered = candidates[
            (candidates['minutes'] <= max_mins) &
            (candidates['n_ingredients'] <= max_ing) &
            (candidates['calories'] <= max_cals)
        ]

        return filtered.sort_values(by='hybrid_score', ascending=False).head(5)
