import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


FEEDBACK_LOG_PATH = "feedback_logs.jsonl"


class FeedbackStore:
    def __init__(self, path=FEEDBACK_LOG_PATH):
        self.path = path
        self.logs = []
        self.load()

    def load(self):
        self.logs = []
        if not os.path.exists(self.path):
            return

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def log(self, user_id, recipe_title, action):
        entry = {
            "user_id": user_id,
            "recipe": recipe_title,
            "action": action,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        self.logs.append(entry)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


memory = FeedbackStore()


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def extract_features(recipe):
    likes = safe_float(recipe.get("likes", 0))
    bookmarks = safe_float(recipe.get("bookmarks", 0))
    popularity = safe_float(recipe.get("popularity_score", 0))
    sustainability = safe_float(recipe.get("sustainability_score", 0.5), 0.5)

    return {
        "popularity": min(popularity / 300.0, 1.0),
        "likes": min(np.log1p(likes) / 8.0, 1.0),
        "bookmarks": min(np.log1p(bookmarks) / 8.0, 1.0),
        "sustainability": sustainability,
    }


def action_to_score(action):
    return {
        "like": 1.0,
        "view": 0.2,
        "dislike": -1.0,
    }.get(action, 0.0)


class SimpleLTR:
    def __init__(self):
        self.weights = {
            "popularity": 0.35,
            "likes": 0.25,
            "bookmarks": 0.20,
            "sustainability": 0.20,
        }
        self.lr = 0.03

    def predict(self, features, feedback_signal):
        base_score = sum(self.weights[key] * features.get(key, 0) for key in self.weights)
        return base_score + 0.3 * feedback_signal

    def update(self, features, feedback_signal, target):
        pred = self.predict(features, feedback_signal)
        error = target - pred

        for key in self.weights:
            self.weights[key] += self.lr * error * features.get(key, 0)

    def train(self, logs, df):
        for log in logs:
            row = df[df["title"] == log["recipe"]]
            if row.empty:
                continue

            features = extract_features(row.iloc[0])
            target = action_to_score(log["action"])
            self.update(features, target, target)


ltr_model = SimpleLTR()


def compute_feedback_signal(recipe_title, user_id=None):
    score = 0.0

    for log in memory.logs:
        if log["recipe"] != recipe_title:
            continue
        if user_id is not None and log["user_id"] != user_id:
            continue
        score += action_to_score(log["action"])

    return score


def rank_ltr(df, user_id=None):
    df = df.copy()
    scores = []

    for _, row in df.iterrows():
        features = extract_features(row)
        feedback_signal = compute_feedback_signal(row["title"], user_id=user_id)
        scores.append(ltr_model.predict(features, feedback_signal))

    df["ltr_score"] = scores
    return df.sort_values("ltr_score", ascending=False)


def update_model(df=None):
    if df is None or len(memory.logs) == 0:
        return

    ltr_model.train(memory.logs, df)


def estimate_food_sustainability(food_name):
    text = str(food_name).lower()

    if any(keyword in text for keyword in ["salad", "rau", "luộc", "hấp", "gỏi", "nấm", "đậu"]):
        return 0.8
    if any(keyword in text for keyword in ["chiên", "thịt", "bò", "heo", "gà"]):
        return 0.55
    if any(keyword in text for keyword in ["cơm", "bún", "phở", "cháo"]):
        return 0.65

    return 0.5


def analyze_meal_plan(meal_plan):
    results = []

    for day in meal_plan:
        for meal, food in day["meals"].items():
            results.append({
                "day": day["day"],
                "meal": meal,
                "food": food,
                "sustainability_score": round(estimate_food_sustainability(food), 2),
            })

    return pd.DataFrame(results)


def print_weights():
    print("\n=== LTR WEIGHTS ===")
    for key, value in ltr_model.weights.items():
        print(f"{key}: {round(value, 4)}")
