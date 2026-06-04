from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import safe_sparse_dot


DEFAULT_NGRAM_RANGE = (3, 5)
DEFAULT_MAX_FEATURES = 100_000
DEFAULT_MIN_DF = 2
DEFAULT_FACTS_MAX_CHARS = 1_200
DEFAULT_OPP_MAX_CHARS = 800


def build_claim_retrieval_text(
    tort: Any,
    party: str,
    claim_idx: int,
    *,
    facts_max_chars: int = DEFAULT_FACTS_MAX_CHARS,
    opp_max_chars: int = DEFAULT_OPP_MAX_CHARS,
) -> str:
    if party == "P":
        claim = tort.plaintiff_claims[claim_idx]
        opposing = tort.defendant_claims
        party_marker = "[原告主張]"
        opp_marker = "[被告反論]"
    else:
        claim = tort.defendant_claims[claim_idx]
        opposing = tort.plaintiff_claims
        party_marker = "[被告主張]"
        opp_marker = "[原告反論]"

    facts = " ".join(item.description for item in tort.undisputed_facts)[:facts_max_chars]
    opp_text = " ".join(item.description for item in opposing)[:opp_max_chars]
    return f"[争いのない事実] {facts} {party_marker} {claim.description} {opp_marker} {opp_text}"


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


def fit_party_retriever(
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


def fit_retriever_bundle(
    train_examples: dict[str, tuple[list[str], list[int]]],
    *,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_features: int = DEFAULT_MAX_FEATURES,
    min_df: int = DEFAULT_MIN_DF,
) -> dict[str, Any]:
    bundles: dict[str, Any] = {}
    for party in ("P", "D"):
        texts, labels = train_examples.get(party, ([], []))
        if texts:
            bundles[party] = fit_party_retriever(
                texts,
                labels,
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
            )
        else:
            bundles[party] = {
                "vectorizer": None,
                "ref_matrix": None,
                "ref_labels": np.zeros(0, dtype=float),
                "positive_rate": 0.5,
            }
    return bundles


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
    if ref_matrix is None or ref_matrix.shape[0] == 0:
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


def predict_party_probs(texts: list[str], party_bundle: dict[str, Any], *, top_k: int) -> np.ndarray:
    if not texts:
        return np.zeros(0, dtype=float)
    vectorizer = party_bundle["vectorizer"]
    if vectorizer is None:
        return np.full(len(texts), float(party_bundle["positive_rate"]), dtype=float)
    query_matrix = vectorizer.transform(texts)
    return predict_topk_probs(
        query_matrix,
        party_bundle["ref_matrix"],
        np.asarray(party_bundle["ref_labels"], dtype=float),
        top_k=int(top_k),
        fallback_prob=float(party_bundle["positive_rate"]),
    )


def predict_retrieval_probs(
    texts_by_party: dict[str, list[str]],
    bundle: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, np.ndarray]:
    return {
        party: predict_party_probs(texts_by_party.get(party, []), bundle[party], top_k=top_k)
        for party in ("P", "D")
    }
