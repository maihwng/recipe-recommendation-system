# 🍲 AI Recipe Recommendation System

Hệ thống gợi ý công thức nấu ăn thông minh dựa trên trí tuệ nhân tạo, cung cấp các đề xuất được cá nhân hóa theo **chỉ số cơ thể (BMI), mục tiêu sức khỏe, sở thích ẩm thực và nguyên liệu sẵn có** thông qua truy vấn ngôn ngữ tự nhiên tiếng Việt.

---

# 📌 Giới thiệu tổng quan

Trong bối cảnh bùng nổ thông tin ẩm thực, người dùng thường gặp khó khăn do **quá tải thông tin** khi lựa chọn món ăn phù hợp với sức khỏe, sở thích cá nhân và nhu cầu dinh dưỡng.

Dự án này xây dựng một hệ thống gợi ý công thức nấu ăn có khả năng **cá nhân hóa sâu sắc**, giúp người dùng tìm kiếm và khám phá món ăn phù hợp dựa trên:

- Thông tin cá nhân.
- Chỉ số cơ thể (BMI).
- Mục tiêu sức khỏe.
- Sở thích ăn uống.
- Nguyên liệu hiện có.
- Truy vấn bằng ngôn ngữ tự nhiên.

Hệ thống không chỉ tìm kiếm công thức dựa trên từ khóa mà còn hiểu được **ngữ nghĩa của câu truy vấn**, từ đó đưa ra các gợi ý phù hợp hơn với nhu cầu thực tế của người dùng.

---

# 🎯 Mục tiêu dự án

- Xây dựng hệ thống Recommendation System cho lĩnh vực ẩm thực.
- Ứng dụng các kỹ thuật NLP để hiểu yêu cầu của người dùng.
- Tìm kiếm công thức dựa trên độ tương đồng ngữ nghĩa.
- Cá nhân hóa kết quả dựa trên thông tin sức khỏe và sở thích.
- Xây dựng pipeline hoàn chỉnh từ thu thập dữ liệu đến triển khai ứng dụng.

---

# ✨ Tính năng chính

## 👤 1. Hồ sơ người dùng & Sức khỏe

Hệ thống xây dựng hồ sơ người dùng dựa trên:

- Chiều cao.
- Cân nặng.
- Chỉ số BMI.
- Mục tiêu dinh dưỡng.

Tự động:
- Tính toán BMI.
- Phân loại tình trạng cơ thể.
- Đề xuất nhóm món ăn phù hợp.

Ví dụ:
```
BMI thấp
→ Gợi ý món giàu năng lượng

BMI cao
→ Gợi ý món ít calo, healthy
```
---

# 🧠 2. Phân tích ý định người dùng (Intent Detection)
Hệ thống sử dụng mô hình Hybrid kết hợp:
- PhoBERT.
- Rule-based System.
- 
Mục tiêu:
- Hiểu được yêu cầu của người dùng từ câu truy vấn tiếng Việt.
- Xác định mục tiêu dinh dưỡng.
- Xác định nhu cầu tìm kiếm món ăn.

Ví dụ:
Input:
```
"Tôi muốn tìm món ăn ít dầu mỡ để giảm cân"
```
Output:
```
Intent:
- Weight Loss

Goal:
- Healthy Food
- Low Calories
```
---

# 🔍 3. Tìm kiếm ngữ nghĩa (Semantic Search)
Thay vì chỉ tìm kiếm dựa trên từ khóa, hệ thống sử dụng:
- Sentence-BERT.
- Vietnamese Sentence-BERT model.
- FAISS (Facebook AI Similarity Search).
```

Điều này giúp hệ thống tìm được các món ăn có **ý nghĩa tương đồng** ngay cả khi câu truy vấn không chứa chính xác từ khóa trong công thức.

---

# 📊 4. Cơ chế xếp hạng (Ranking Engine)

Sau bước truy xuất ứng viên, hệ thống thực hiện xếp hạng dựa trên nhiều tiêu chí:

- Độ tương đồng ngữ nghĩa.
- Mục tiêu sức khỏe.
- Sở thích ẩm thực.
- Thành phần nguyên liệu.
- Độ phổ biến của món ăn.

Công thức tổng hợp điểm:

Final Score = Semantic Similarity + Health Goal Matching + Preference Score + Popularity Score
---

# 💡 5. Giải thích kết quả (Explainable AI - XAI)
Hệ thống cung cấp lý do tại sao món ăn được đề xuất.
Ví dụ:
```
Món ăn này được đề xuất vì:
✓ Phù hợp mục tiêu giảm cân
✓ Hàm lượng calo thấp
✓ Giàu protein
✓ Có nguyên liệu bạn yêu thích
```
Giúp tăng:
- Tính minh bạch.
- Độ tin cậy.
- Trải nghiệm người dùng.
---
# 📅 6. Lập kế hoạch bữa ăn (Meal Planner)
Hệ thống có khả năng tự động tạo thực đơn:
- 7 ngày trong tuần.
- Đảm bảo đa dạng món ăn.
- Phù hợp mục tiêu dinh dưỡng.
Ví dụ:
```
Monday:
- Breakfast
- Lunch
- Dinner
Tuesday:
- Breakfast
- Lunch
- Dinner
...
```
---

# 🔄 7. Học từ phản hồi người dùng (Learning-to-Rank)
Hệ thống cải thiện chất lượng gợi ý thông qua hành vi người dùng:
- Like.
- View.
- Dislike.

Các phản hồi được sử dụng để:
- Điều chỉnh trọng số xếp hạng.
- Cải thiện độ chính xác đề xuất.
- Cá nhân hóa theo thời gian.
---
# 🛠 Công nghệ sử dụng
## Programming Language
- Python
## User Interface
- Streamlit
## Natural Language Processing
- PhoBERT
- Sentence-BERT
- Vietnamese SBERT (`keepitreal/vietnamese-sbert`)
## Vector Search
- FAISS
(Facebook AI Similarity Search)
## Data Collection
- Selenium
- BeautifulSoup
Số lượng dữ liệu:
```
~15,000 công thức nấu ăn
```
## Data Processing
- Pandas
- NumPy
- Scikit-learn

# 📊 Kết quả thực nghiệm
Hệ thống được đánh giá trong môi trường thử nghiệm và đạt được các kết quả
| Metric | Result |
|---|---:|
| Goal Accuracy | 79.9% |
| Recall@10 | 82.6% |
| NDCG@10 | 84.2% |
| Retrieval Latency | ~4.6 ms |
| CTR Lift | 11.02% |
---

# 📈 Đánh giá kết quả
Kết quả thực nghiệm cho thấy hệ thống:
- Hiểu được ý định của người dùng thông qua truy vấn tiếng Việt.
- Tìm kiếm công thức dựa trên ngữ nghĩa thay vì chỉ dựa vào từ khóa.
- Cá nhân hóa kết quả theo mục tiêu sức khỏe.
- Có tốc độ truy xuất gần thời gian thực.
- Có khả năng cải thiện thông qua phản hồi người dùng.
---
Một số hướng phát triển:

- Fine-tune Sentence-BERT trên dữ liệu công thức tiếng Việt.
- Xây dựng chatbot hỗ trợ nấu ăn.
- Kết hợp Large Language Model (LLM).
- Cải thiện Learning-to-Rank.
- Xây dựng hệ thống triển khai thực tế trên nền tảng Web/Mobile.
---
