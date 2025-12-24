import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.title("📘 Exam Score Prediction System")
st.write("Upload dataset and enter student details")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Exam_Score_Prediction.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the dataset to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

# Drop unnecessary column
df = df.drop("student_id", axis=1)

# Split features & target
X = df.drop("exam_score", axis=1)
y = df["exam_score"]

# Categorical & Numerical columns
categorical_cols = [
    "gender", "course", "internet_access",
    "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty"
]

numerical_cols = [
    "age", "study_hours",
    "class_attendance", "sleep_hours"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Model
model = LinearRegression()

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train model
pipeline.fit(X, y)

st.success("Dataset loaded & model trained successfully ✅")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", 10, 40, 20)
gender = st.selectbox("Gender", ["male", "female", "other"])
course = st.selectbox(
    "Course",
    ["bca", "b.sc", "diploma", "b.tech", "bba", "ba", "b.com"]
)
study_hours = st.number_input("Study Hours per Day", 0.0, 15.0, 5.0)
class_attendance = st.number_input("Class Attendance (%)", 0.0, 100.0, 75.0)
internet_access = st.selectbox("Internet Access", ["yes", "no"])
sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
study_method = st.selectbox(
    "Study Method",
    ["self study", "coaching", "online videos", "group study", "mixed"]
)
facility_rating = st.selectbox("Facility Rating", ["low", "medium", "high"])
exam_difficulty = st.selectbox("Exam Difficulty", ["easy", "moderate", "hard"])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Exam Score"):
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": class_attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
    }])

    predicted_score = pipeline.predict(input_data)[0]

    if predicted_score < 50:
        performance = "❌ Poor"
    elif predicted_score < 75:
        performance = "⚠️ Average"
    else:
        performance = "✅ Good"

    st.success(f"🎯 Predicted Exam Score: {predicted_score:.2f}")
    st.info(f"📊 Performance Level: {performance}")
