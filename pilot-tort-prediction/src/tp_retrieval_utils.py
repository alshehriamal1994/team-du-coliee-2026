from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import safe_sparse_dot


DEFAULT_NGRAM_RANGE = (3, 5)
DEFAULT_MAX_FEATURES = 120_000
DEFAULT_MIN_DF = 2
DEFAULT_TEXT_MAX_CHARS = 2_400


def build_retrieval_case_text(tort: Any, *, max_chars: int = DEFAULT_TEXT_MAX_CHARS) -> str:
    uu = " ".join(item.description for item in tort.undisputed_facts)
    pp = " ".join(item.description for item in tort.plaintiff_claims)
    dd = " ".join(item.description for item in tort.defendant_claims)
    text = f"[UU] {uu} [PP] {pp} [DD] {dd}"
    return text[:max_chars]


def build_vectorizer(
    *,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_features: int = DEFAULT_MAX_FEATURES,
    min_df: int = DEFAULT_MIN_DF,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def fit_retriever_bundle(
    texts: list[str],
    labels: list[int] | np.ndarray,
    *,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_features: int = DEFAULT_MAX_FEATURES,
    min_df: int = DEFAULT_MIN_DF,
) -> dict[str, Any]:
    vectorizer = build_vectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )
    ref_matrix = vectorizer.fit_transform(texts)
    ref_labels = np.asarray(labels, dtype=float)
    positive_rate = float(ref_labels.mean()) if len(ref_labels) else 0.5
    return {
        "vectorizer": vectorizer,
        "ref_matrix": ref_matrix,
        "ref_labels": ref_labels,
        "positive_rate": positive_rate,
    }


def predict_topk_probs(
    query_matrix,
    ref_matrix,
    ref_labels: np.ndarray,
    *,
    top_k: int,
    fallback_prob: float,
) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if ref_matrix.shape[0] == 0:
        return np.full(query_matrix.shape[0], float(fallback_prob), dtype=float)

    sims = safe_sparse_dot(query_matrix, ref_matrix.T, dense_output=True)
    probs = np.empty(query_matrix.shape[0], dtype=float)
    ref_count = ref_matrix.shape[0]
    k = min(int(top_k), ref_count)

    for row_idx in range(query_matrix.shape[0]):
        row = np.asarray(sims[row_idx], dtype=float)
        if row.ndim != 1:
            row = row.ravel()
        if k == ref_count:
            candidate_idx = np.arange(ref_count)
        else:
            candidate_idx = np.argpartition(row, -k)[-k:]
        weights = np.clip(row[candidate_idx], 0.0, None)
        if float(weights.sum()) <= 0.0:
            probs[row_idx] = float(fallback_prob)
            continue
        probs[row_idx] = float(np.dot(weights, ref_labels[candidate_idx]) / weights.sum())
    return probs


def extract_topk_feature_matrix(
    query_matrix,
    ref_matrix,
    ref_labels: np.ndarray,
    *,
    top_k: int,
    fallback_prob: float,
) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    ref_labels = np.asarray(ref_labels, dtype=float)
    if ref_matrix.shape[0] == 0:
        return np.full((query_matrix.shape[0], 4), float(fallback_prob), dtype=float)

    sims = safe_sparse_dot(query_matrix, ref_matrix.T, dense_output=True)
    ref_count = ref_matrix.shape[0]
    k = min(int(top_k), ref_count)
    features = np.zeros((query_matrix.shape[0], 4 + (2 * int(top_k))), dtype=float)

    for row_idx in range(query_matrix.shape[0]):
        row = np.asarray(sims[row_idx], dtype=float)
        if row.ndim != 1:
            row = row.ravel()
        if k == ref_count:
            candidate_idx = np.arange(ref_count)
        else:
            candidate_idx = np.argpartition(row, -k)[-k:]
        order = np.argsort(row[candidate_idx])[::-1]
        top_idx = candidate_idx[order]
        top_sims = np.clip(row[top_idx], 0.0, None)
        top_labels = ref_labels[top_idx]

        sim_sum = float(top_sims.sum())
        weighted_prob = float(np.dot(top_sims, top_labels) / sim_sum) if sim_sum > 0.0 else float(fallback_prob)
        top1_sim = float(top_sims[0]) if len(top_sims) >= 1 else 0.0
        top2_sim = float(top_sims[1]) if len(top_sims) >= 2 else 0.0
        sim_gap = top1_sim - top2_sim

        features[row_idx, 0] = weighted_prob
        features[row_idx, 1] = sim_sum
        features[row_idx, 2] = top1_sim
        features[row_idx, 3] = sim_gap

        for j in range(int(top_k)):
            base = 4 + (2 * j)
            if j < len(top_sims):
                features[row_idx, base] = float(top_sims[j])
                features[row_idx, base + 1] = float(top_labels[j])
            else:
                features[row_idx, base] = 0.0
                features[row_idx, base + 1] = float(fallback_prob)
    return features


def predict_retrieval_probs(
    texts: list[str],
    bundle: dict[str, Any],
    *,
    top_k: int,
) -> np.ndarray:
    if not texts:
        return np.zeros(0, dtype=float)
    query_matrix = bundle["vectorizer"].transform(texts)
    return predict_topk_probs(
        query_matrix,
        bundle["ref_matrix"],
        np.asarray(bundle["ref_labels"], dtype=float),
        top_k=int(top_k),
        fallback_prob=float(bundle["positive_rate"]),
    )


def predict_retrieval_feature_matrix(
    texts: list[str],
    bundle: dict[str, Any],
    *,
    top_k: int,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 4 + (2 * int(top_k))), dtype=float)
    query_matrix = bundle["vectorizer"].transform(texts)
    return extract_topk_feature_matrix(
        query_matrix,
        bundle["ref_matrix"],
        np.asarray(bundle["ref_labels"], dtype=float),
        top_k=int(top_k),
        fallback_prob=float(bundle["positive_rate"]),
    )


def build_reranker_feature_matrix(
    texts: list[str],
    bundle: dict[str, Any],
    base_probs: np.ndarray,
    *,
    top_k: int,
    k_feature: int,
) -> np.ndarray:
    base_probs = np.asarray(base_probs, dtype=float)
    if len(texts) != len(base_probs):
        raise ValueError("texts and base_probs must have the same length.")
    if not texts:
        return np.zeros((0, 1 + 4 + (2 * int(k_feature))), dtype=float)
    query_matrix = bundle["vectorizer"].transform(texts)
    agg = extract_topk_feature_matrix(
        query_matrix,
        bundle["ref_matrix"],
        np.asarray(bundle["ref_labels"], dtype=float),
        top_k=int(top_k),
        fallback_prob=float(bundle["positive_rate"]),
    )[:, :1]
    slots = extract_topk_feature_matrix(
        query_matrix,
        bundle["ref_matrix"],
        np.asarray(bundle["ref_labels"], dtype=float),
        top_k=int(k_feature),
        fallback_prob=float(bundle["positive_rate"]),
    )
    return np.concatenate([base_probs[:, None], agg, slots[:, 1:]], axis=1)


def blend_case_probs(base_probs: np.ndarray, retrieval_probs: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    return ((1.0 - alpha) * np.asarray(base_probs, dtype=float)) + (alpha * np.asarray(retrieval_probs, dtype=float))
