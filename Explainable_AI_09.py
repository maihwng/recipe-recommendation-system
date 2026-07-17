import json

import pandas as pd


def load_user(user_id):
    with open(f"user_profiles/{user_id}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def explain_recipe(breakdown):
    explanations = [
        f"Độ giống ngữ nghĩa: {breakdown['similarity']:.2f}",
        f"Khớp mục tiêu cá nhân: {breakdown['goal']:.2f}",
        f"Khớp nguyên liệu: {breakdown['ingredient']:.2f}",
        f"Khớp ẩm thực yêu thích: {breakdown['cuisine']:.2f}",
        f"Độ phổ biến: {breakdown['popularity']:.2f}",
    ]

    if "keyword" in breakdown:
        explanations.append(f"Khớp từ khóa: {breakdown['keyword']:.2f}")
    if "title_keyword" in breakdown:
        explanations.append(f"Từ khóa nằm trong tên món: {breakdown['title_keyword']:.2f}")

    if breakdown.get("title_keyword", 0) > 0:
        explanations.append("Món này được ưu tiên vì tên món khớp trực tiếp với từ khóa bạn nhập.")
    elif breakdown.get("keyword", 0) > 0:
        explanations.append("Món này có nguyên liệu hoặc mô tả khớp với từ khóa bạn nhập.")
    elif breakdown["similarity"] > 0.7:
        explanations.append("Món này gần nghĩa với nhu cầu tìm kiếm của bạn.")

    return explanations


def explain_ranked_list(scored_list, top_k=10):
    results = []

    for item in scored_list[:top_k]:
        recipe = item["recipe"]
        breakdown = item["breakdown"]

        results.append({
            "title": str(recipe.get("title", "")),
            "image_url": str(recipe.get("image_url", "")),
            "ingredients": str(recipe.get("ingredients", "")),
            "instructions": str(recipe.get("instructions", "")),
            "likes": int(recipe.get("likes", 0) if pd.notna(recipe.get("likes")) else 0),
            "score": item["score"],
            "search_score": item.get("search_score", 0),
            "personal_score": item.get("personal_score", 0),
            "relevance_score": item.get("relevance_score", 0),
            "explanation": explain_recipe(breakdown),
            "breakdown": breakdown,
        })

    return results


def print_explanations(explained_list):
    for item in explained_list:
        print("\n" + "=" * 40)
        print("RECIPE:", item["title"])
        print("SCORE:", round(item["score"], 3))

        for explanation in item["explanation"]:
            print("-", explanation)
