import json

import numpy as np


def load_user(user_id):
    with open(f"user_profiles/{user_id}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_query(profile, intent, raw_query=""):
    query_parts = []

    if raw_query:
        query_parts.append(raw_query)

    query_parts.append(profile.get("goal", ""))
    query_parts.append(profile.get("preferred_cuisine", ""))
    query_parts += profile.get("dietary_restrictions", [])
    query_parts.append(intent.get("intent", ""))
    query_parts.append(intent.get("meal_type", ""))
    query_parts += intent.get("ingredients", [])
    query_parts += intent.get("flavors", [])
    query_parts += intent.get("cooking_methods", [])

    return " ".join(part for part in query_parts if part)


def encode_query(query, model):
    return model.encode([query])[0].astype("float32")


def search_faiss(query_vector, index, top_k=300):
    query_vector = np.array([query_vector])
    distances, indices = index.search(query_vector, top_k)
    return indices[0], distances[0]


def get_candidates(indices, distances, df):
    candidates = df.iloc[indices].copy().reset_index(drop=True)
    candidates["distance"] = distances
    return candidates


def keyword_match_level(row, raw_keywords):
    title = str(row.get("title", "")).lower()
    ingredients = str(row.get("ingredients", "")).lower()
    keywords = str(row.get("keywords", "")).lower()
    text = f"{title} {ingredients} {keywords}"

    if any(keyword in title for keyword in raw_keywords):
        return 2
    if any(keyword in text for keyword in raw_keywords):
        return 1
    return 0


def add_keyword_match_level(df, raw_keywords):
    result = df.copy()
    result["keyword_match_level"] = result.apply(
        lambda row: keyword_match_level(row, raw_keywords), axis=1
    )
    return result


def keyword_filter(candidates, intent, min_results=5):
    raw_keywords = intent.get("raw_keywords", [])
    if not raw_keywords:
        return candidates

    candidates = add_keyword_match_level(candidates, raw_keywords)
    filtered = candidates[candidates["keyword_match_level"] > 0]

    if len(filtered) >= min_results:
        return filtered.reset_index(drop=True)

    return candidates.reset_index(drop=True)


def direct_keyword_search(df, intent, min_results=10):
    raw_keywords = intent.get("raw_keywords", [])
    if not raw_keywords:
        return None

    matched = add_keyword_match_level(df, raw_keywords)
    matched = matched[matched["keyword_match_level"] > 0].copy()

    if len(matched) < min_results:
        return None

    matched["distance"] = matched["keyword_match_level"].map({2: 0.01, 1: 0.2})
    return matched.reset_index(drop=True)


def candidate_search(user_id, intent, model, index, df, raw_query=""):
    profile = load_user(user_id)

    direct_matches = direct_keyword_search(df, intent)
    if direct_matches is not None:
        return direct_matches

    query = build_query(profile, intent, raw_query)
    query_vec = encode_query(query, model)
    indices, distances = search_faiss(query_vec, index, top_k=300)
    candidates = get_candidates(indices, distances, df)

    return keyword_filter(candidates, intent)
