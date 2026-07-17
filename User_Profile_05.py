import json
import os
from datetime import datetime


PROFILE_DIR = "user_profiles"


def calculate_bmi(weight, height):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)


def classify_bmi_vn(bmi):
    if bmi < 18.5:
        return "Gầy"
    if bmi <= 22.9:
        return "Bình thường"
    if bmi <= 24.9:
        return "Thừa cân"
    if bmi <= 29.9:
        return "Béo phì độ I"
    return "Béo phì độ II"


def ideal_weight(height_cm):
    return {
        "ideal_weight": round((height_cm - 100) * 0.9, 1),
        "min_weight": round((height_cm - 100) * 0.8, 1),
        "max_weight": round(height_cm - 100, 1),
    }


def create_user_profile(user_id, age, gender, weight, height,
                        goal, preferred_cuisine, dietary_restrictions):
    bmi = calculate_bmi(weight, height)

    return {
        "user_id": user_id,
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "goal": goal,
        "preferred_cuisine": preferred_cuisine,
        "dietary_restrictions": dietary_restrictions,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": {
            "bmi": bmi,
            "bmi_label": classify_bmi_vn(bmi),
            "ideal_weight": ideal_weight(height),
        },
    }


def save_user_profile(profile):
    os.makedirs(PROFILE_DIR, exist_ok=True)
    path = os.path.join(PROFILE_DIR, f"{profile['user_id']}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=4)


def load_user_profile(user_id):
    path = os.path.join(PROFILE_DIR, f"{user_id}.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
