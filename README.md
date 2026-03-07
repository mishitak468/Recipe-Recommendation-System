# AI Kitchen Scout: Personalized Recipe Recommendation System

A sophisticated, data-driven recommendation engine built with **Python**, **Pandas**, and **Scikit-learn**. This system leverages the Food.com (GeniusKitchen) dataset, featuring over 180k recipes and 700k user interactions. It utilizes a hybrid approach to suggest meals based on ingredient similarity, user ratings, and nutritional constraints.

## Key Features

* **Hybrid Recommendation Engine**: Combines **Content-Based Filtering** (TF-IDF Vectorization & Cosine Similarity) with **Collaborative Filtering** (Average User Ratings) to rank suggestions.
* **"Pantry Mode" Search**: Allows users to input available ingredients to find matching recipes using vector space modeling.
* **Advanced Complexity Filters**: Users can filter results by preparation time (`minutes`) and ingredient count (`n_ingredients`).
* **Nutritional Optimization**: Integrated filtering for caloric limits based on recipe metadata.
* **Interactive Streamlit UI**: A professional web interface for real-time recipe discovery and exploration.



## Technical Stack

* **Language**: Python 3.9+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (TfidfVectorizer, Cosine Similarity)
* **Frontend**: Streamlit
* **Version Control**: Git

## Project Structure

```text
recipe-recommender/
├── data/               # Local storage for RAW_recipes.csv & RAW_interactions.csv
├── src/
│   ├── app.py          # Streamlit web application logic
│   ├── engine.py       # Recommendation algorithms and data processing
│   └── __init__.py     # Package initialization
├── .gitignore          # Configured to ignore large datasets and venv
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
