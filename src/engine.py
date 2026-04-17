from __future__ import annotations
import ast
import time
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Helpers

def _parse_list(x: str) -> list:
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _ndcg_at_k(relevance: list, k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    relevance = relevance[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _reciprocal_rank(relevance: list) -> float:
    """Mean Reciprocal Rank helper — returns 1/(rank of first relevant item)."""
    for i, rel in enumerate(relevance):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


# MMR reranking
def _mmr_rerank(
    candidates_df: pd.DataFrame,
    tfidf_matrix,
    candidate_indices: list,
    lambda_mmr: float = 0.6,
    k: int = 5,
) -> list:
    """
    Maximal Marginal Relevance: balances relevance vs. intra-list diversity.
    lambda_mmr=1.0 → pure relevance; lambda_mmr=0.0 → pure diversity.
    """
    selected = []
    remaining = list(candidate_indices)

    while len(selected) < k and remaining:
        best_idx = None
        best_score = -np.inf

        for idx in remaining:
            relevance = candidates_df.loc[idx, "hybrid_score"]
            if selected:
                sim_to_selected = max(
                    cosine_similarity(
                        tfidf_matrix[idx], tfidf_matrix[s]
                    )[0][0]
                    for s in selected
                )
            else:
                sim_to_selected = 0.0

            score = lambda_mmr * relevance - (1 - lambda_mmr) * sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# Main recommender
class RecipeRecommender:
    def __init__(self, recipe_path: str, sample_size: int = 15000):
        t0 = time.perf_counter()

        self.df = pd.read_csv(recipe_path).head(
            sample_size).reset_index(drop=True)

        # --- Parsing ---
        self.df["ingredient_list"] = self.df["ingredients"].apply(_parse_list)
        self.df["clean_ingredients"] = self.df["ingredient_list"].apply(
            lambda lst: " ".join(i.replace(" ", "") for i in lst)
        )
        self.df["ingredient_set"] = self.df["ingredient_list"].apply(
            lambda lst: {i.lower().strip() for i in lst}
        )
        self.df["calories"] = self.df["nutrition"].apply(
            lambda x: _parse_list(x)[0] if _parse_list(x) else 0
        )

        # TF-IDF
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.df["clean_ingredients"])
        self.cosine_sim = cosine_similarity(
            self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(
            self.df.index, index=self.df["name"]).drop_duplicates()

        # Tracking for coverage
        self._recommendation_counts = np.zeros(len(self.df))

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"RecipeRecommender loaded {len(self.df)} recipes in {elapsed:.1f}ms")

    # Retrieval helpers

    def _tfidf_sim_scores(self, title: str) -> np.ndarray:
        idx = self.indices[title]
        return self.cosine_sim[idx]

    def _pantry_sim_scores(self, ingredients_input: str) -> np.ndarray:
        query = ingredients_input.replace(",", " ").lower()
        query_vec = self.tfidf.transform([query])
        return cosine_similarity(query_vec, self.tfidf_matrix).flatten()

    def _jaccard_sim_scores(self, query_set: set) -> np.ndarray:
        return np.array([_jaccard(query_set, s) for s in self.df["ingredient_set"]])

    def _bayesian_smooth_ratings(
        self, candidates: pd.DataFrame, global_mean: float, m: int = 10
    ) -> pd.Series:
        """
        Bayesian average: rating = (m * global_mean + sum_ratings) / (m + n_ratings)
        Eliminates bias where a recipe with 1 perfect rating beats one with 500 good ratings.
        m = minimum number of reviews needed to be 'trusted'.
        """
        return candidates.apply(
            lambda row: (m * global_mean +
                         row["rating"] * row.get("review_count", 1))
            / (m + row.get("review_count", 1)),
            axis=1,
        )

    # Core recommend
    def recommend(
        self,
        title: str | None,
        interaction_path: str,
        max_mins: int = 60,
        max_ing: int = 10,
        max_cals: float = 500,
        pantry_input: str | None = None,
        k: int = 5,
        lambda_mmr: float = 0.6,
        alpha: float = 0.7,        # TF-IDF weight (1-alpha = Jaccard weight)
        tfidf_weight: float = 0.7,  # content vs. rating split
    ) -> tuple[pd.DataFrame | None, dict]:
        """
        Returns (results_df, metrics_dict).
        metrics_dict contains Precision@K, Recall@K, NDCG@K, MRR, Coverage, Diversity, Novelty.
        """
        timings: dict[str, float] = {}
        stage = time.perf_counter()

        # Similarity scores
        query_set: set = set()

        if pantry_input:
            tfidf_scores = self._pantry_sim_scores(pantry_input)
            query_set = {i.strip().lower() for i in pantry_input.split(",")}
            jaccard_scores = self._jaccard_sim_scores(query_set)
        elif title and title in self.indices:
            tfidf_scores = self._tfidf_sim_scores(title)
            idx = self.indices[title]
            query_set = self.df.loc[idx, "ingredient_set"]
            jaccard_scores = self._jaccard_sim_scores(query_set)
        else:
            logger.warning(f"Title '{title}' not found. Returning None.")
            return None, {}

        # Fused content score
        content_scores = alpha * tfidf_scores + (1 - alpha) * jaccard_scores
        timings["retrieval_ms"] = (time.perf_counter() - stage) * 1000

        # Load interactions + Bayesian ratings
        stage = time.perf_counter()
        interactions = pd.read_csv(interaction_path)
        agg = interactions.groupby("recipe_id")["rating"].agg(
            rating="mean", review_count="count"
        ).reset_index()
        global_mean = agg["rating"].mean()

        candidates = self.df.merge(
            agg, left_on="id", right_on="recipe_id", how="left"
        ).fillna({"rating": global_mean, "review_count": 0})
        timings["ratings_load_ms"] = (time.perf_counter() - stage) * 1000

        # Hybrid score with Bayesian smoothing
        stage = time.perf_counter()
        candidates["content_score"] = content_scores
        candidates["smoothed_rating"] = self._bayesian_smooth_ratings(
            candidates, global_mean)
        candidates["hybrid_score"] = (
            tfidf_weight * candidates["content_score"]
            + (1 - tfidf_weight) * (candidates["smoothed_rating"] / 5.0)
        )
        timings["scoring_ms"] = (time.perf_counter() - stage) * 1000

        # Hard filters
        filtered = candidates[
            (candidates["minutes"] <= max_mins)
            & (candidates["n_ingredients"] <= max_ing)
            & (candidates["calories"] <= max_cals)
        ].copy()

        if title:
            # Exclude the query recipe itself
            filtered = filtered[filtered["name"] != title]

        if filtered.empty:
            return pd.DataFrame(), {}

        # MMR reranking
        stage = time.perf_counter()
        filtered = filtered.reset_index(drop=True)
        # top-50 to keep it fast
        pool = filtered.nlargest(min(50, len(filtered)), "hybrid_score")
        pool_indices = list(pool.index)
        selected_indices = _mmr_rerank(
            pool, self.tfidf_matrix, pool_indices, lambda_mmr=lambda_mmr, k=k
        )
        results = pool.loc[selected_indices].copy()
        timings["mmr_ms"] = (time.perf_counter() - stage) * 1000
        timings["total_ms"] = sum(timings.values())

        # Update popularity tracking
        for orig_idx in results["id"].values:
            mask = self.df["id"] == orig_idx
            self._recommendation_counts[mask] += 1

        # Compute
        metrics = self._compute_metrics(
            results=results,
            filtered_pool=filtered,
            candidates=candidates,
            k=k,
            timings=timings,
        )

        return results, metrics

    # Metrics
    def _compute_metrics(
        self,
        results: pd.DataFrame,
        filtered_pool: pd.DataFrame,
        candidates: pd.DataFrame,
        k: int,
        timings: dict,
    ) -> dict:
        if results.empty:
            return {}

        # Precision@K & Recall@K
        # "relevant" if its rating >= 4.0
        RELEVANCE_THRESHOLD = 4.0
        relevance_flags = (results["rating"] >=
                           RELEVANCE_THRESHOLD).astype(int).tolist()

        precision_at_k = sum(relevance_flags[:k]) / k
        total_relevant_in_pool = (
            filtered_pool["rating"] >= RELEVANCE_THRESHOLD).sum()
        recall_at_k = (
            sum(relevance_flags[:k]) / total_relevant_in_pool
            if total_relevant_in_pool > 0 else 0.0
        )

        # NDCG@K
        graded = (results["rating"] / 5.0).tolist()
        ndcg = _ndcg_at_k(graded, k)

        # MRR
        mrr = _reciprocal_rank(relevance_flags)

        # Coverage
        n_recommended_ever = int((self._recommendation_counts > 0).sum())
        catalogue_coverage = n_recommended_ever / len(self.df)

        # ILD
        result_indices = list(results.index)
        if len(result_indices) > 1:
            pairwise_sims = []
            for i in range(len(result_indices)):
                for j in range(i + 1, len(result_indices)):
                    sim = cosine_similarity(
                        self.tfidf_matrix[result_indices[i]],
                        self.tfidf_matrix[result_indices[j]],
                    )[0][0]
                    pairwise_sims.append(sim)
            ild = 1 - np.mean(pairwise_sims)
        else:
            ild = 0.0

        # Lower popularity = higher novelty
        counts = self._recommendation_counts[result_indices]
        novelty = float(np.mean(-np.log2((counts + 1) / (len(self.df) + 1))))

        # Filter pass-rate
        filter_pass_rate = len(filtered_pool) / max(len(candidates), 1)

        return {
            "precision_at_k": round(precision_at_k, 4),
            "recall_at_k": round(recall_at_k, 4),
            "ndcg_at_k": round(ndcg, 4),
            "mrr": round(mrr, 4),
            "catalogue_coverage": round(catalogue_coverage, 4),
            "intra_list_diversity": round(ild, 4),
            "novelty": round(novelty, 4),
            "filter_pass_rate": round(filter_pass_rate, 4),
            "results_returned": len(results),
            "k": k,
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        }

    # Offline evaluation harness

    def evaluate_offline(
        self, interaction_path: str, n_test_users: int = 100, k: int = 5
    ) -> dict:
        """
        Simulate leave-one-out evaluation:
        For each of n_test_users, hold out their last rated recipe,
        recommend from the rest, and check if the held-out item appears.
        Returns macro-averaged Precision@K, Recall@K, NDCG@K, MRR.
        """
        interactions = pd.read_csv(interaction_path)
        user_groups = interactions.groupby("user_id")
        users = list(user_groups.groups.keys())[:n_test_users]

        all_metrics: list[dict] = []

        for user_id in users:
            user_df = user_groups.get_group(user_id).sort_values("date") \
                if "date" in interactions.columns \
                else user_groups.get_group(user_id)

            if len(user_df) < 2:
                continue

            held_out_id = user_df.iloc[-1]["recipe_id"]
            held_out_row = self.df[self.df["id"] == held_out_id]
            if held_out_row.empty:
                continue

            title = held_out_row.iloc[0]["name"]
            results, metrics = self.recommend(
                title=title,
                interaction_path=interaction_path,
                k=k,
            )

            if results is not None and not results.empty:
                hit = int(held_out_id in results["id"].values)
                metrics["hit"] = hit
                all_metrics.append(metrics)

        if not all_metrics:
            return {}

        agg = {
            "macro_precision_at_k": float(np.mean([m["precision_at_k"] for m in all_metrics])),
            "macro_recall_at_k": float(np.mean([m["recall_at_k"] for m in all_metrics])),
            "macro_ndcg_at_k": float(np.mean([m["ndcg_at_k"] for m in all_metrics])),
            "macro_mrr": float(np.mean([m["mrr"] for m in all_metrics])),
            "hit_rate_at_k": float(np.mean([m["hit"] for m in all_metrics])),
            "n_evaluated": len(all_metrics),
        }
        logger.info(f"Offline eval complete: {agg}")
        return agg

    def metrics_report(self, last_metrics: dict) -> str:
        """Human-readable summary of the last call's metrics."""
        if not last_metrics:
            return "No metrics available."
        lines = [
            f"Precision@{last_metrics['k']}: {last_metrics['precision_at_k']:.2%}",
            f"Recall@{last_metrics['k']}:    {last_metrics['recall_at_k']:.2%}",
            f"NDCG@{last_metrics['k']}:      {last_metrics['ndcg_at_k']:.4f}",
            f"MRR:              {last_metrics['mrr']:.4f}",
            f"ILD (diversity):  {last_metrics['intra_list_diversity']:.4f}",
            f"Novelty:          {last_metrics['novelty']:.4f}",
            f"Coverage:         {last_metrics['catalogue_coverage']:.2%}",
            f"Filter pass rate: {last_metrics['filter_pass_rate']:.2%}",
            f"Latency:          {last_metrics['timings_ms']['total_ms']:.1f}ms total",
            f"  ├ retrieval:    {last_metrics['timings_ms']['retrieval_ms']:.1f}ms",
            f"  ├ ratings load: {last_metrics['timings_ms']['ratings_load_ms']:.1f}ms",
            f"  ├ scoring:      {last_metrics['timings_ms']['scoring_ms']:.1f}ms",
            f"  └ MMR rerank:   {last_metrics['timings_ms']['mmr_ms']:.1f}ms",
        ]
        return "\n".join(lines)
