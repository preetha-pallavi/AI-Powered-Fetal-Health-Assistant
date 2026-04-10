import streamlit as st
import joblib
import numpy as np
import pandas as pd
from Fetal_health_LLM import explain_prediction, ask_followup

# 1. Load model and scaler
model = joblib.load("fetal_health_model.pkl")
scaler = joblib.load("fetal_scaler.pkl")

# 2. App title
st.title("🏥 Fetal Health Assistant")
st.write("Enter CTG values to assess fetal health.")

# 3. File upload or manual input
st.subheader("Input Method")
uploaded_file = st.file_uploader("Upload a CTG CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    # Read first row automatically
    baseline = df["baseline value"].iloc[0]
    accelerations = df["accelerations"].iloc[0]
    fetal_movement = df["fetal_movement"].iloc[0]
    uterine_contractions = df["uterine_contractions"].iloc[0]
    light_decelerations = df["light_decelerations"].iloc[0]
    severe_decelerations = df["severe_decelerations"].iloc[0]
    prolongued_decelerations = df["prolongued_decelerations"].iloc[0]
    abnormal_short_term_variability = df["abnormal_short_term_variability"].iloc[0]
    mean_value_of_short_term_variability = df["mean_value_of_short_term_variability"].iloc[0]
    percentage_of_time_with_abnormal_long_term_variability = df["percentage_of_time_with_abnormal_long_term_variability"].iloc[0]
    mean_value_of_long_term_variability = df["mean_value_of_long_term_variability"].iloc[0]

else:
    # Manual input
    baseline = st.number_input("Baseline Fetal Heart Rate", value=120.0)
    accelerations = st.number_input("Accelerations", value=0.003)
    fetal_movement = st.number_input("Fetal Movement", value=0.0)
    uterine_contractions = st.number_input("Uterine Contractions", value=0.003)
    light_decelerations = st.number_input("Light Decelerations", value=0.003)
    severe_decelerations = st.number_input("Severe Decelerations", value=0.0)
    prolongued_decelerations = st.number_input("Prolongued Decelerations", value=0.0)
    abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", value=73.0)
    mean_value_of_short_term_variability = st.number_input("Mean Short Term Variability", value=0.5)
    percentage_of_time_with_abnormal_long_term_variability = st.number_input("% Time Abnormal Long Term Variability", value=43.0)
    mean_value_of_long_term_variability = st.number_input("Mean Long Term Variability", value=2.4)

# 4. Histogram values — fixed dataset averages (not shown to user)
histogram_width = 70.0
histogram_min = 60.0
histogram_max = 160.0
histogram_number_of_peaks = 2.0
histogram_number_of_zeroes = 0.0
histogram_tendency = 0.0
histogram_variance = 70.0

# 5. Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "prediction_context" not in st.session_state:
    st.session_state.prediction_context = ""
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

# 6. Predict button
if st.button("Analyse"):
    input_data = np.array([[baseline, accelerations, fetal_movement,
                            uterine_contractions, light_decelerations,
                            severe_decelerations, prolongued_decelerations,
                            abnormal_short_term_variability,
                            mean_value_of_short_term_variability,
                            percentage_of_time_with_abnormal_long_term_variability,
                            mean_value_of_long_term_variability,
                            histogram_width, histogram_min, histogram_max,
                            histogram_number_of_peaks, histogram_number_of_zeroes,
                            histogram_tendency, histogram_variance]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0]

    class_map = {1: "🟢 Normal", 2: "🟡 Suspect", 3: "🔴 Pathological"}

    with st.spinner("Generating explanation..."):
        explanation = explain_prediction(int(prediction), confidence)

    # Store everything in session state
    st.session_state.prediction = class_map[int(prediction)]
    st.session_state.confidence = confidence
    st.session_state.explanation = explanation
    st.session_state.prediction_done = True
    st.session_state.history = []
    st.session_state.last_answer = ""
    st.session_state.prediction_context = f"""
    Previous CTG Analysis Result:
    - Prediction: {class_map[int(prediction)]}
    - Confidence — Normal: {confidence[0]*100:.1f}%, Suspect: {confidence[1]*100:.1f}%, Pathological: {confidence[2]*100:.1f}%
    - LLM Explanation: {explanation}
    """

# 7. Show prediction results (persists after button clicks)
if st.session_state.prediction_done:
    st.subheader(f"Prediction: {st.session_state.prediction}")
    st.write(
        f"Confidence — Normal: {st.session_state.confidence[0]*100:.1f}% | "
        f"Suspect: {st.session_state.confidence[1]*100:.1f}% | "
        f"Pathological: {st.session_state.confidence[2]*100:.1f}%"
    )
    st.info(st.session_state.explanation)

    # 8. Follow-up chat
    st.subheader("💬 Ask a follow-up question")
    question = st.text_input("Your question")
    if st.button("Ask"):
        if question.strip() != "":
            full_question = f"Context: {st.session_state.prediction_context}\n\nQuestion: {question}"
            answer, updated_history = ask_followup(full_question, st.session_state.history)
            st.session_state.history = updated_history
            st.session_state.last_answer = answer

    if st.session_state.last_answer != "":
        st.write(f"**Answer:** {st.session_state.last_answer}")

# 9. Disclaimer
st.markdown("---")
st.caption("⚠️ This tool is for clinical assistance only and does not replace professional medical diagnosis.")