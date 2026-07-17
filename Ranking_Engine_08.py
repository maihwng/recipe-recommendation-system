import json

import numpy as np
import pandas as pd


def load_user(user_id):
    with open(f"user_profiles/{user_id}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def goal_score(recipe, profile):
    goal = profile.get("goal", "")
    title = str(recipe.get("title", "")).lower()

    if goal == "weight_loss":
        if any(keyword in title for keyword in ["salad", "luộc", "hấp", "healthy", "gỏi", "rau"]):
            return 1.0
    elif goal == "weight_gain":
        if any(keyword in title for keyword in ["cơm", "chiên", "xào", "thịt", "gà", "bò"]):
            return 1.0
    elif goal == "healthy":
        if any(keyword in title for keyword in ["salad", "rau", "luộc", "hấp", "healthy", "gỏi"]):
            return 1.0

    return 0.5


def ingredient_score(recipe, intent):
    ingredients = intent.get("ingredients", [])
    if not ingredients:
        return 0.5

    recipe_text = (
        str(recipe.get("title", "")) + " " +
        str(recipe.get("ingredients", ""))
    ).lower()
    matched = sum(1 for ingredient in ingredients if ingredient.lower() in recipe_text)
    return matched / len(ingredients)


def cuisine_score(recipe, profile):
    cuisine = profile.get("preferred_cuisine", "").lower()
    recipe_cuisine = str(recipe.get("cuisine", "")).lower()

    if cuisine and cuisine in recipe_cuisine:
        return 1.0

    return 0.5


def similarity_score(distance):
    return 1 / (1 + float(distance))


def popularity_score(recipe):
    try:
        likes = float(recipe.get("likes", 0))
        bookmarks = float(recipe.get("bookmarks", 0))
    except Exception:
        likes = 0
        bookmarks = 0

    score = 0.7 * np.log1p(likes) + 0.3 * np.log1p(bookmarks)
    return min(score / 10.0, 1.0)


def keyword_score(recipe, intent):
    raw_keywords = intent.get("raw_keywords", [])
    if not raw_keywords:
        return 0.5

    recipe_text = (
        str(recipe.get("title", "")) + " " +
        str(recipe.get("ingredients", "")) + " " +
        str(recipe.get("keywords", ""))
    ).lower()
    matched = sum(1 for keyword in raw_keywords if keyword in recipe_text)
    return matched / len(raw_keywords)


def title_keyword_score(recipe, intent):
    raw_keywords = intent.get("raw_keywords", [])
    if not raw_keywords:
        return 0.0

    title = str(recipe.get("title", "")).lower()
    matched = sum(1 for keyword in raw_keywords if keyword in title)
    return matched / len(raw_keywords)


def compute_scores(recipe, profile, intent):
    sim = similarity_score(recipe.get("distance", 1.0))
    goal = goal_score(recipe, profile)
    ingredient = ingredient_score(recipe, intent)
    cuisine = cuisine_score(recipe, profile)
    popularity = popularity_score(recipe)
    keyword = keyword_score(recipe, intent)
    title_keyword = max(
        float(recipe.get("keyword_match_level", 0)) / 2.0,
        title_keyword_score(recipe, intent),
    )

    search_score = 0.45 * keyword + 0.35 * title_keyword + 0.20 * sim

    if intent.get("raw_keywords", []):
        relevance_score = search_score
        personal_score = 0.50 * goal + 0.30 * ingredient + 0.20 * cuisine
        final_score = 0.70 * relevance_score + 0.20 * personal_score + 0.10 * popularity
    else:
        relevance_score = sim
        personal_score = 0.45 * goal + 0.30 * ingredient + 0.25 * cuisine
        final_score = 0.45 * relevance_score + 0.35 * personal_score + 0.20 * popularity

    return {
        "score": final_score,
        "search_score": search_score,
        "personal_score": personal_score,
        "relevance_score": relevance_score,
        "breakdown": {
            "similarity": sim,
            "goal": goal,
            "ingredient": ingredient,
            "cuisine": cuisine,
            "popularity": popularity,
            "keyword": keyword,
            "title_keyword": title_keyword,
        },
    }


def rank_candidates(candidates, profile, intent):
    scored = []

    for _, row in candidates.iterrows():
        recipe = row.to_dict()
        scores = compute_scores(recipe, profile, intent)
        scored.append({
            "recipe": recipe,
            **scores,
        })

    return sorted(scored, key=lambda item: item["score"], reverse=True)


def get_top_k(scored_list, k=10):
    return pd.DataFrame([item["recipe"] for item in scored_list[:k]])


def ranking_engine(user_id, candidates, intent):
    profile = load_user(user_id)
    scored = rank_candidates(candidates, profile, intent)
    top10 = get_top_k(scored)
    return top10, scored
