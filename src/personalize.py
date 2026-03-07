import pandas as pd


def get_user_top_recipe(user_id, interaction_path, recipe_path):
    interactions = pd.read_csv(interaction_path)
    recipes = pd.read_csv(recipe_path)

    # Find the highest-rated recipe for this user
    user_ratings = interactions[interactions['user_id'] == user_id]
    if user_ratings.empty:
        return None

    best_recipe_id = user_ratings.sort_values(
        by='rating', ascending=False).iloc[0]['recipe_id']

    # Get the name of that recipe
    recipe_name = recipes[recipes['id'] == best_recipe_id].iloc[0]['name']
    return recipe_name


if __name__ == "__main__":
    # Example User ID from the dataset
    UID = 38094
    fav = get_user_top_recipe(
        UID, 'data/RAW_interactions.csv', 'data/RAW_recipes.csv')
    print(f"User {UID}'s favorite recipe: {fav}")
