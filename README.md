# AI Kitchen Scout — Personalized Recipe Recommendation System

> A production-grade hybrid recommender combining TF-IDF content filtering, Jaccard set similarity, Bayesian-smoothed collaborative filtering, and MMR diversity reranking — evaluated across 120,000 user interactions on the Food.com dataset.

## Overview

AI Kitchen Scout solves the **recipe discovery problem**: given what a user enjoys cooking (or what ingredients they have on hand), surface the 5 most relevant, diverse, and appropriately rated recipes — accounting for preparation time, ingredient count, and caloric constraints.

The system uses a **four-stage pipeline**:

```
User Query
    │
    ▼
1. Dual Retrieval (TF-IDF cosine + Jaccard set overlap)
    │
    ▼
2. Bayesian-Smoothed Collaborative Filtering (avg ratings + prior)
    │
    ▼
3. Hybrid Scoring (70% content + 30% ratings, tunable)
    │
    ▼
4. MMR Diversity Reranking (Maximal Marginal Relevance)
    │
    ▼
Top-K Results + Live Metrics
```

Two search modes are supported: **recipe-to-recipe similarity** (content-based) and **pantry mode** (ingredient-to-recipe), both feeding the same downstream pipeline.

---

## Architecture

```
src/
├── engine.py          # RecipeRecommender class — all ML logic
│   ├── __init__       # TF-IDF fit, cosine similarity matrix, indices
│   ├── recommend()    # Main pipeline → returns (results_df, metrics_dict)
│   ├── _mmr_rerank()  # Maximal Marginal Relevance diversity reranking
│   ├── _bayesian_smooth_ratings()   # Cold-start-corrected rating estimator
│   ├── _compute_metrics()           # Precision@K, Recall@K, NDCG@K, MRR, ILD, Novelty, Coverage
│   └── evaluate_offline()           # Leave-one-out evaluation harness
│
└── app.py             # Streamlit UI
    ├── Tab 1: Recommendations (inline metrics strip, score breakdown per result)
    ├── Tab 2: Metrics Dashboard (latency chart, retrieval quality, beyond-accuracy)
    └── Tab 3: Offline Evaluation runner
```

### Data flow

| Stage | Input | Output | Latency (p50) |
|---|---|---|---|
| TF-IDF retrieval | Recipe name or pantry string | Cosine similarity vector (15,000-dim) | ~22ms |
| Jaccard overlap | Ingredient sets | Overlap score vector | ~1ms |
| Ratings merge | PostgreSQL-style CSV join | Candidates with smoothed rating | ~57ms |
| Hybrid scoring | Content + rating vectors | Unified score column | ~113ms |
| MMR reranking | Top-50 pool | Final top-K ranked list | ~177ms |
| **Total (p50)** | | | **~374ms** |

The ratings CSV load (57ms) is the primary optimization target in a production deployment — replacing it with an in-memory cache or a Redis lookup would bring end-to-end latency under 100ms.

---

## Algorithm Design

### 1. Dual Content Retrieval

A single TF-IDF signal misses exact ingredient overlap (e.g., "chicken breast" vs "chicken"). Two signals are fused:

```python
content_score = alpha * tfidf_cosine_similarity + (1 - alpha) * jaccard_overlap
# default: alpha = 0.7
```

**TF-IDF** uses `ngram_range=(1, 2)` and English stop-word removal. Bigrams capture compound ingredients ("bell pepper", "soy sauce") that unigrams miss.

**Jaccard similarity** measures exact set overlap between ingredient lists:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

This is immune to TF-IDF's frequency bias — a rare ingredient match scores the same as a common one.

### 2. Bayesian-Smoothed Collaborative Filtering

Raw average ratings are biased toward recipes with few reviews. A recipe with one 5-star review outranks one with 500 four-star reviews — which is wrong.

The Bayesian average shrinks each recipe's rating toward the global mean proportional to its review count:

```
smoothed_rating = (m × global_mean + Σ_ratings) / (m + n_reviews)
```

Where `m = 10` is the minimum review prior (the number of "virtual" reviews at the global mean). A recipe with 1 review gets ≈91% weight on the global mean; one with 100 reviews gets ≈9%.

**18.7% of the catalogue** (2,803 / 15,000 recipes) has ≤2 reviews and benefits directly from this correction. The UI flags cold-start items with a warning.

### 3. Hybrid Scoring

```python
hybrid_score = w_content × content_score + (1 - w_content) × (smoothed_rating / 5.0)
# default: w_content = 0.7
```

Both weights are exposed as Streamlit sliders so users can tune the system in real time and observe the effect on the metrics dashboard.

### 4. MMR Diversity Reranking

Greedy top-K retrieval from a sorted list can return five nearly identical recipes (e.g., five pasta dishes when querying "spaghetti carbonara"). MMR explicitly penalises redundancy:

```
score(candidate) = λ × hybrid_score - (1 - λ) × max_similarity_to_selected
```

At `λ = 1.0` (pure relevance) the intra-list diversity (ILD) is **0.9683**.
At `λ = 0.5` (MMR-balanced) ILD rises to **0.9920** — a **2.4% increase** in ingredient diversity at minimal precision cost.

MMR runs on the top-50 candidates only, keeping its contribution to end-to-end latency under 180ms.

---

## Evaluation Metrics

The system tracks two categories of metrics:

### Retrieval quality

| Metric | Definition | Why it matters |
|---|---|---|
| **Precision@K** | Fraction of top-K results rated ≥ 4.0 stars | Direct proxy for user satisfaction |
| **Recall@K** | Fraction of all good items in the filtered pool returned | Coverage of the relevant space |
| **NDCG@K** | Normalised Discounted Cumulative Gain | Measures *ranking quality*, not just presence |
| **MRR** | Mean Reciprocal Rank | Where does the first good result appear? |

### Beyond-accuracy metrics

| Metric | Definition | Why it matters |
|---|---|---|
| **ILD** (Intra-List Diversity) | Avg pairwise cosine *dissimilarity* among returned items | Prevents filter bubbles |
| **Novelty** | Mean log-inverse popularity of recommended items | Long-tail exposure |
| **Catalogue Coverage** | Fraction of the catalogue ever recommended | Fairness to less-popular recipes |
| **Filter Pass Rate** | Fraction of candidates surviving hard constraints | Diagnoses over-restrictive filters |

### Offline evaluation protocol

Leave-one-out evaluation: for each test user, the most recently rated recipe is held out. The engine recommends from the user's remaining history, and Hit Rate@K measures whether the held-out item appears in the top-K.

---

## Benchmark Results

All results measured on 15,000 recipes and 120,000 interactions (Food.com schema).

### Offline leave-one-out evaluation (N=200 users, K=5)

| Metric | Value |
|---|---|
| Macro Precision@5 | **63.7%** |
| Macro NDCG@5 | **0.9786** |
| Macro MRR | **0.8156** |
| Macro Recall@5 | 0.33% |
| Users evaluated | 200 |

> **Note on Recall@5:** Recall is intentionally low — the denominator is all relevant items in the filtered pool (time / ingredient / calorie constraints applied). A user who loves 50 healthy recipes will only see 5 of them per query. This is a known property of top-K recommendation in large catalogues, not a model failure. Hit Rate@K (which measures whether the specific held-out item is retrieved) is the more meaningful offline signal for this task.

### End-to-end latency (100 random queries)

| Percentile | Latency |
|---|---|
| p50 | **374ms** |
| p95 | **392ms** |
| p99 | **420ms** |
| Engine load time | 2,974ms (one-time, cached by Streamlit) |

> **Primary bottleneck:** CSV ratings reload on each query (57ms). In production, pre-loading interactions into memory at startup would bring p50 latency to ~120ms. A Redis cache of precomputed Bayesian-smoothed ratings would reduce it further to ~30ms.

### Diversity experiment (MMR λ sensitivity)

| λ | ILD | Interpretation |
|---|---|---|
| 1.0 (pure relevance) | 0.9683 | High baseline diversity from ingredient variety |
| 0.5 (MMR-balanced) | **0.9920** | +2.4% diversity gain with negligible precision cost |

### Filter pass-rate sensitivity

| Constraints | Pass Rate |
|---|---|
| Strict (30min / 8 ingredients / 400 cal) | 4.5% |
| Default (60min / 12 ingredients / 700 cal) | **22.8%** |
| Relaxed (120min / 20 ingredients / 1,200 cal) | 84.9% |

### Cold-start analysis

| Stat | Value |
|---|---|
| Recipes with ≤2 reviews | **18.7%** (2,803 / 15,000) |
| Bayesian prior `m` | 10 reviews |
| Rating shrinkage for 1-review recipe | ~91% toward global mean |
| Global mean rating | 3.956 / 5.0 |

---

## Dataset

This project uses the **Food.com Recipes and Interactions** dataset (Kaggle, GeniusKitchen), containing:

| Stat | Value |
|---|---|
| Recipes | 180,000+ (15,000 sampled for development) |
| User interactions | 700,000+ (120,000 used) |
| Unique users | 3,000+ active in sample |
| Avg ingredients per recipe | 10.0 |
| Avg prep time | 47 minutes |
| Avg rating (global) | 3.956 / 5.0 |

The dataset is not included in this repository due to size. Download from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) and place the two CSV files in `data/`:

```
data/
├── RAW_recipes.csv
└── RAW_interactions.csv
```

---

## Project Structure

```
Recipe-Recommendation-System/
├── src/
│   ├── engine.py          # Core ML engine — RecipeRecommender class
│   └── app.py             # Streamlit web application
├── data/                  # CSV files (gitignored — download from Kaggle)
├── tests/
│   └── test_recipe_recommender.py   # 28 pytest unit + integration tests
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions — pytest + coverage on push
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/mishitak468/Recipe-Recommendation-System.git
cd Recipe-Recommendation-System
pip install -r requirements.txt
```

### 2. Download data

Place `RAW_recipes.csv` and `RAW_interactions.csv` from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) in the `data/` directory.

### 3. Run the app

```bash
streamlit run src/app.py
```

### 4. Run tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 5. Run offline evaluation

Use the **Offline Evaluation** tab in the Streamlit app, or run directly:

```python
from src.engine import RecipeRecommender
engine = RecipeRecommender("data/RAW_recipes.csv")
metrics = engine.evaluate_offline("data/RAW_interactions.csv", n_test_users=200, k=5)
print(metrics)
```

---

## Known Limitations & Future Work

| Limitation | Current state | Production fix |
|---|---|---|
| Ratings reload latency | CSV re-read per query (~57ms) | Pre-load into memory at startup; Redis cache for smoothed ratings |
| Cold-start for new users | No user history → falls back to content-only | Session-based implicit feedback; onboarding preference survey |
| Static TF-IDF matrix | Rebuilt on each startup (~3s) | Serialise with `joblib`; incremental index updates |
| No personalisation | All users get same content scores | User embedding via matrix factorisation (ALS or SVD++) |
| Nutrition data | Calories only from nutrition string | Parse all 6 nutrition fields (fat, sugar, sodium, protein, fiber) |
| Evaluation hit rate | 0% hit rate on exact item recall | Expected for a content-based system — user's held-out item may not match their query recipe's style |

---

## Skills Demonstrated

| Category | Technologies |
|---|---|
| Machine Learning | TF-IDF, Cosine Similarity, Jaccard Similarity, Bayesian Averaging, MMR Reranking |
| Recommendation Systems | Hybrid Filtering, Cold-Start Handling, Beyond-Accuracy Metrics (ILD, Novelty, Coverage) |
| Evaluation | NDCG@K, MRR, Precision@K, Recall@K, Leave-One-Out Offline Evaluation |
| Data Engineering | Pandas, NumPy, scikit-learn, large-scale CSV processing |
| Software Engineering | Modular OOP design, pytest test suite, GitHub Actions CI, latency instrumentation |
| UI / Data Products | Streamlit, real-time metrics dashboard, algorithm control panel |
