# Recipe-Recommendation-System

A high-performance, data-driven recommendation engine built with Python, Pandas, and Scikit-learn. This project utilizes the Food.com (GeniusKitchen) dataset—comprising 180k+ recipes and 700k+ user interactions—to deliver tailored meal suggestions based on ingredient similarity, user ratings, and nutritional constraints.

**Key Features**
- Hybrid Recommendation Engine: Combines Content-Based Filtering (via TF-IDF Vectorization and Cosine Similarity) with Collaborative Filtering (average user ratings) to rank suggestions.
- Pantry Mode (Inverse Search): Allows users to input specific ingredients they have on hand (e.g., "chicken, garlic, spinach") to find matching recipes.
- Complexity and Health Filters: Integrated sliders to filter results by maximum preparation time, number of ingredients, and caloric limits.
- Advanced Data Parsing: Engineered custom parsers to transform raw string-based metadata (ingredients and nutritional lists) into structured numerical features.
- Interactive Streamlit UI: A professional web interface for real-time querying and recipe exploration.

**Tech Stack**
- Language: Python 3.9+
- Libraries: Pandas, NumPy, Scikit-learn
- Frontend: Streamlit
- Data Source: Kaggle: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
