import pandas as pd


MEALS = ["breakfast", "lunch", "dinner"]


def filter_by_goal(df, goal):
    if goal == "weight_loss":
        pattern = "salad|luộc|hấp|healthy|gỏi|rau"
        filtered = df[df["title"].str.contains(pattern, case=False, na=False)]
    elif goal == "weight_gain":
        pattern = "cơm|chiên|xào|thịt|gà|bò"
        filtered = df[df["title"].str.contains(pattern, case=False, na=False)]
    elif goal == "healthy":
        pattern = "salad|rau|hấp|luộc|yến mạch|healthy|gỏi|nấm"
        filtered = df[df["title"].str.contains(pattern, case=False, na=False)]
    else:
        filtered = df

    if filtered.empty:
        filtered = df

    return filtered.sort_values("likes", ascending=False).head(150)


def generate_daily_plan(df_filtered):
    day_plan = {}

    for meal in MEALS:
        if df_filtered.empty:
            day_plan[meal] = "No data"
            continue

        day_plan[meal] = df_filtered.sample(1).iloc[0]["title"]

    return day_plan


def generate_week_plan(user_id, load_user, df):
    profile = load_user(user_id)
    goal = profile.get("goal", "healthy")
    df_filtered = filter_by_goal(df, goal)

    week_plan = []
    used = set()

    for day in range(7):
        day_plan = {}

        for meal in MEALS:
            options = df_filtered[~df_filtered["title"].isin(list(used))]
            if options.empty:
                options = df_filtered

            choice = options.sample(1).iloc[0]
            used.add(choice["title"])
            day_plan[meal] = choice["title"]

        week_plan.append({
            "day": f"Day {day + 1}",
            "meals": day_plan,
        })

    return week_plan


def print_week_plan(plan):
    for day in plan:
        print("\n====================")
        print(day["day"])

        for meal, food in day["meals"].items():
            print(f"{meal}: {food}")
