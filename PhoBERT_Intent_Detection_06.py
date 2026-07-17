import re


LABELS = ["weight_loss", "weight_gain", "healthy"]

GOAL_KEYWORDS = {
    "weight_loss": [
        "giảm cân", "diet", "low cal", "eat clean", "healthy",
        "không béo", "fit body", "lean", "salad", "luộc", "hấp", "gỏi",
        "rau", "nhẹ", "ít calo", "giảm", "không mỡ"
    ],
    "weight_gain": [
        "tăng cân", "bulk", "protein", "mass",
        "béo", "mập", "cơm", "chiên", "xào", "thịt", "bò", "gà",
        "mỡ", "nước mắm", "dầu", "nướng", "tăng", "nhiều"
    ],
    "healthy": [
        "healthy", "cân bằng", "balance", "lành mạnh",
        "dinh dưỡng", "rau", "nấm", "yến mạch", "đậu phụ",
        "sức khỏe", "fit", "wellness", "tươi", "nguyên chất"
    ],
}

MEAL_KEYWORDS = {
    "breakfast": ["sáng", "morning", "breakfast", "bữa sáng"],
    "lunch": ["trưa", "lunch", "bữa trưa"],
    "dinner": ["tối", "evening", "dinner", "bữa tối"],
}

INGREDIENTS = [
    "gà", "bò", "tôm", "cá", "trứng", "thịt", "heo", "lợn", "mực", "cua",
    "ốc", "nghêu", "hến", "rau", "cơm", "khoai", "sữa", "yến mạch",
    "hành", "tỏi", "ớt", "tiêu", "nước mắm", "đậu hũ", "đậu phụ",
    "nấm", "bắp", "ngô", "cà rốt", "cà chua", "bí đỏ", "su hào",
    "cải", "măng", "bún", "phở", "miến", "mì",
]

FLAVOR_KEYWORDS = [
    "cay", "ngọt", "mặn", "chua", "béo", "thanh", "nhạt", "đắng", "thơm",
]

COOKING_METHOD_KEYWORDS = [
    "chiên", "luộc", "hấp", "xào", "nướng", "kho", "nấu", "rang",
    "trộn", "gỏi", "hầm", "om", "lẩu", "phi",
]

user_memory = {}


def clean_text(text: str):
    text = str(text).lower()
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rule_based_intent(text: str):
    text = clean_text(text)

    # Score each goal by keyword match count
    goal_scores = {}
    for label, keywords in GOAL_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        goal_scores[label] = score
    
    # Pick goal with highest score, default to healthy
    if max(goal_scores.values()) > 0:
        goal = max(goal_scores, key=goal_scores.get)
    else:
        goal = "healthy"

    meal_type = "any"
    for label, keywords in MEAL_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            meal_type = label
            break

    ingredients = [item for item in INGREDIENTS if item in text]
    flavors = [item for item in FLAVOR_KEYWORDS if item in text]
    cooking_methods = [item for item in COOKING_METHOD_KEYWORDS if item in text]

    return {
        "intent": goal,
        "meal_type": meal_type,
        "ingredients": ingredients,
        "flavors": flavors,
        "cooking_methods": cooking_methods,
        "raw_keywords": flavors + cooking_methods + ingredients,
    }


INTENT_SEEDS = {
    "weight_loss": "giảm cân không béo fit body lean diet low calorie eat clean salad luộc hấp gỏi rau",
    "weight_gain": "tăng cân béo mập protein bulk mass cơm chiên xào thịt bò gà nướng",
    "healthy": "healthy sức khỏe cân bằng lành mạnh dinh dưỡng rau nấm yến mạch đậu phụ wellness fit",
}


def embedding_predict(text: str, model):
    text_emb = model.encode(text)
    scores = {}

    for label, seed in INTENT_SEEDS.items():
        seed_emb = model.encode(seed)
        denom = (text_emb @ text_emb) ** 0.5 * (seed_emb @ seed_emb) ** 0.5
        scores[label] = float((text_emb @ seed_emb) / denom) if denom else 0.0

    best_label = max(scores, key=scores.get)
    return {"intent": best_label, "confidence": scores[best_label]}


def get_user_history(user_id):
    return user_memory.get(user_id, [])


def update_user_history(user_id, query, intent):
    user_memory.setdefault(user_id, []).append({
        "query": query,
        "intent": intent,
    })


def analyze_query(query: str, model, user_id: str = None, threshold: float = 0.40):
    rule_result = rule_based_intent(query)
    emb_result = embedding_predict(query, model)

    if emb_result["confidence"] >= threshold:
        method = "embedding"
        intent = {
            **rule_result,
            "intent": emb_result["intent"],
        }
        confidence = emb_result["confidence"]
    else:
        method = "rule_based"
        intent = rule_result
        confidence = 0.0

    if user_id:
        history = get_user_history(user_id)
        if history and intent["intent"] == "healthy" and not intent["raw_keywords"]:
            intent["intent"] = history[-1]["intent"].get("intent", "healthy")
        update_user_history(user_id, query, intent)

    return {
        "query": query,
        "method": method,
        "confidence": round(confidence, 4),
        "intent": intent["intent"],
        "meal_type": intent["meal_type"],
        "ingredients": intent.get("ingredients", []),
        "flavors": intent.get("flavors", []),
        "cooking_methods": intent.get("cooking_methods", []),
        "raw_keywords": intent.get("raw_keywords", []),
    }
