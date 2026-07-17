import os
import re

import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from Candidate_Search_07 import candidate_search
from Explainable_AI_09 import explain_ranked_list
from Meal_Planning_10 import generate_week_plan
from PhoBERT_Intent_Detection_06 import analyze_query
from Ranking_Engine_08 import ranking_engine
from Sustainability_11 import analyze_meal_plan, memory, update_model
from User_Profile_05 import create_user_profile, load_user_profile, save_user_profile


PROFILE_DIR = "user_profiles"
os.makedirs(PROFILE_DIR, exist_ok=True)

st.set_page_config(page_title="AI Food System", layout="wide")


@st.cache_resource
def load_models_and_index():
    model = SentenceTransformer("keepitreal/vietnamese-sbert")
    index = faiss.read_index("recipe.index")
    return model, index


@st.cache_data
def load_dataset():
    return pd.read_csv("clean_recipe_dataset.csv")


def split_instruction_steps(instructions):
    text = str(instructions or "").strip()
    if not text or text.lower() == "nan":
        return []

    if "|" in text:
        parts = text.split("|")
    else:
        parts = re.split(r"(?:\n+)|(?:\s+\d+[\.\)]\s+)", text)

    return [part.strip(" -\t\r\n") for part in parts if part.strip(" -\t\r\n")]


with st.spinner("Đang tải dữ liệu và mô hình AI..."):
    model, index = load_models_and_index()
    df = load_dataset()


if "results" not in st.session_state:
    st.session_state.results = None
if "scored" not in st.session_state:
    st.session_state.scored = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "guest"
if "meal_plan" not in st.session_state:
    st.session_state.meal_plan = None


st.title("AI Nutrition Recommender System")

tab_profile, tab_recommend, tab_meal, tab_feedback = st.tabs([
    "User Profile",
    "Recommendation",
    "Meal Planner",
    "Feedback & Sustainability",
])


with tab_profile:
    st.header("Tạo / Cập nhật User Profile")

    user_id_input = st.text_input("Tên tài khoản", st.session_state.user_id)
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Tuổi", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
        weight = st.number_input("Cân nặng (kg)", min_value=30.0, max_value=200.0, value=60.0)
        height = st.number_input("Chiều cao (cm)", min_value=100.0, max_value=220.0, value=165.0)

    with col2:
        goal = st.selectbox("Mục tiêu", ["Bình thường", "healthy", "weight_loss", "weight_gain"])
        cuisine = st.text_input("Ẩm thực yêu thích", "vietnamese")
        diet = st.text_input("Kiêng ăn, cách nhau bằng dấu phẩy", "")

    if st.button("Lưu Profile"):
        st.session_state.user_id = user_id_input
        diet_list = [item.strip() for item in diet.split(",") if item.strip()]

        profile = create_user_profile(
            user_id=st.session_state.user_id,
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            goal=goal,
            preferred_cuisine=cuisine,
            dietary_restrictions=diet_list,
        )
        save_user_profile(profile)
        st.success("Đã lưu profile!")
        st.metric("BMI", profile["features"]["bmi"])
        st.info(profile["features"]["bmi_label"])


with tab_recommend:
    st.subheader("Tìm món ăn thông minh")

    query = st.text_input("Nhập món ăn / nguyên liệu / nhu cầu", "Tôi muốn ăn gì đó healthy")

    if st.button("Gợi ý"):
        try:
            load_user_profile(st.session_state.user_id)
        except Exception:
            st.error("Profile chưa tồn tại. Vui lòng tạo profile ở tab User Profile trước.")
            st.stop()

        intent = analyze_query(query, model, user_id=st.session_state.user_id)
        candidates = candidate_search(st.session_state.user_id, intent, model, index, df, raw_query=query)
        top10, scored = ranking_engine(st.session_state.user_id, candidates, intent)
        explained = explain_ranked_list(scored)

        st.session_state.results = top10
        st.session_state.scored = explained
        st.success("Đã tạo danh sách gợi ý!")

    if st.session_state.scored is not None:
        st.write("### Top món ăn gợi ý")

        for item in st.session_state.scored[:10]:
            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                if item.get("image_url") and str(item["image_url"]) != "nan":
                    st.image(item["image_url"])
                else:
                    st.info("Không có ảnh")

            with col2:
                st.subheader(item.get("title", "Món ăn"))
                st.write(f"**Lượt thích:** {item.get('likes', 0)}")

                score_col1, score_col2, score_col3 = st.columns(3)
                score_col1.metric("Điểm tổng hợp", round(item.get("score", 0), 3))
                score_col2.metric("Khớp tìm kiếm", round(item.get("search_score", 0), 3))
                score_col3.metric("Cá nhân hóa", round(item.get("personal_score", 0), 3))

                with st.expander("Nguyên liệu"):
                    st.write(item.get("ingredients", "Không có thông tin nguyên liệu"))
                with st.expander("Các bước nấu"):
                    steps = split_instruction_steps(item.get("instructions", ""))
                    if steps:
                        for step_number, step in enumerate(steps, start=1):
                            st.markdown(f"**Bước {step_number}:** {step}")
                    else:
                        st.write("Không có thông tin các bước nấu")
                with st.expander("Giải thích gợi ý"):
                    for explanation in item["explanation"]:
                        st.write(f"- {explanation}")

                st.write("---")
                st.write("**Phản hồi của bạn giúp AI học tốt hơn:**")
                fb_col1, fb_col2, fb_col3 = st.columns(3)
                recipe_title = item.get("title", "Unknown")

                with fb_col1:
                    if st.button("Thích", key=f"like_{recipe_title}"):
                        memory.log(st.session_state.user_id, recipe_title, "like")
                        update_model(df)
                        st.success("Đã ghi nhận!")
                with fb_col2:
                    if st.button("Đã xem", key=f"view_{recipe_title}"):
                        memory.log(st.session_state.user_id, recipe_title, "view")
                        update_model(df)
                        st.success("Đã ghi nhận!")
                with fb_col3:
                    if st.button("Không thích", key=f"dislike_{recipe_title}"):
                        memory.log(st.session_state.user_id, recipe_title, "dislike")
                        update_model(df)
                        st.success("Đã ghi nhận!")


MEAL_LABEL = {
    "breakfast": "Bữa sáng",
    "lunch": "Bữa trưa",
    "dinner": "Bữa tối",
}


with tab_meal:
    st.subheader("Gợi ý thực đơn tuần")

    try:
        profile = load_user_profile(st.session_state.user_id)
        st.info(
            f"Thực đơn được tạo theo mục tiêu **{profile.get('goal', '')}** "
            f"của tài khoản **{st.session_state.user_id}**."
        )
    except Exception:
        st.warning("Chưa có profile. Vui lòng tạo profile ở tab User Profile trước.")

    if st.button("Tạo thực đơn 7 ngày"):
        try:
            plan = generate_week_plan(st.session_state.user_id, load_user_profile, df)
            st.session_state.meal_plan = plan
            st.success("Đã tạo thực đơn theo mục tiêu của bạn!")
        except Exception as exc:
            st.error(f"Lỗi tạo thực đơn. Có thể user chưa có profile. Chi tiết: {exc}")

    if st.session_state.meal_plan is not None:
        for day in st.session_state.meal_plan:
            with st.expander(day["day"], expanded=True):
                for meal, food in day["meals"].items():
                    st.write(f"{MEAL_LABEL.get(meal, meal)}: **{food}**")


with tab_feedback:
    st.subheader("Phân tích & tối ưu hệ thống")
    st.write(f"Feedback đã lưu: **{len(memory.logs)}** lượt")

    if st.session_state.meal_plan is not None:
        result_df = analyze_meal_plan(st.session_state.meal_plan)
        st.write("### Sustainability Score của thực đơn")
        st.dataframe(result_df)
        score = result_df["sustainability_score"].mean()
        st.metric("Overall Sustainability", round(score, 2))
    else:
        st.info("Vui lòng tạo thực đơn ở tab Meal Planner để phân tích.")
