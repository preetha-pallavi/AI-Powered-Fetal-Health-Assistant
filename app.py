import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from Fetal_health_LLM import explain_prediction, ask_followup

# 1. Load model and scaler
model = joblib.load("fetal_health_model.pkl")
scaler = joblib.load("fetal_scaler.pkl")

# 2. Feature status chart
def plot_feature_status(input_values):
    ranges = {
        "Baseline FHR": (110, 160, input_values["baseline"]),
        "Accelerations": (0.001, 0.02, input_values["accelerations"]),
        "Fetal Movement": (0.001, 0.05, input_values["fetal_movement"]),
        "Uterine Contractions": (0.001, 0.015, input_values["uterine_contractions"]),
        "Light Decelerations": (0.0, 0.005, input_values["light_decelerations"]),
        "Severe Decelerations": (0.0, 0.001, input_values["severe_decelerations"]),
        "Prolongued Decelerations": (0.0, 0.001, input_values["prolongued_decelerations"]),
        "Abnormal STV": (0, 70, input_values["abnormal_stv"]),
        "% Abnormal LTV": (0, 50, input_values["pct_abnormal_ltv"]),
        "Mean LTV": (2, 10, input_values["mean_ltv"]),
    }

    features = []
    values = []
    colors = []
    statuses = []

    for feature, (low, high, value) in ranges.items():
        features.append(feature)
        values.append(value)

        if low <= value <= high:
            colors.append("#2ecc71")
            statuses.append("🟢 Normal")
        elif value < low:
            colors.append("#f39c12")
            statuses.append("🟡 Low")
        else:
            colors.append("#e74c3c")
            statuses.append("🔴 Concerning")

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=statuses,
        textposition="outside"
    ))

    fig.update_layout(
        title="CTG Feature Status",
        xaxis_title="Value",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        showlegend=False
    )

    return fig

# 3. App title
st.title("🏥 Fetal Health Assistant")
st.write("Enter CTG values to assess fetal health.")

# 4. About ranges expander
with st.expander("ℹ️ About the normal ranges used"):
    st.markdown("""
    The colour-coded chart uses the following reference ranges:

    | Feature | Normal Range | Source |
    |---|---|---|
    | Baseline FHR | 110–160 bpm | FIGO Guidelines |
    | Accelerations | > 0.001 | FIGO Guidelines |
    | Prolongued Decelerations | 0.0–0.001 | FIGO Guidelines |
    | Severe Decelerations | 0.0–0.001 | FIGO Guidelines |
    | Light Decelerations | 0.0–0.005 | Dataset distribution |
    | Uterine Contractions | 0.001–0.015 | Dataset distribution |
    | Abnormal STV | 0–70 | Dataset distribution |
    | % Abnormal LTV | 0–50 | Dataset distribution |
    | Mean LTV | 2–10 | Dataset distribution |

    **Sources:**
    - [FIGO Intrapartum Fetal Monitoring Guidelines](https://www.figo.org/resources/figo-guidelines)
    - [WHO recommendations for fetal monitoring](https://www.who.int/publications/i/item/9789241550215)
    - Dataset: [Kaggle — Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
    """)

# 5. File upload or manual input
st.subheader("Input Method")
uploaded_file = st.file_uploader("Upload a CTG CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

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
    with st.expander("❤️ Heart Rate", expanded=True):
        baseline = st.slider("Baseline Fetal Heart Rate (bpm)", 50, 200, 120)
        accelerations = st.slider("Accelerations", 0.0, 0.05, 0.003, step=0.001)
        fetal_movement = st.slider("Fetal Movement", 0.0, 0.05, 0.0, step=0.001)

    with st.expander("📉 Decelerations", expanded=True):
        uterine_contractions = st.slider("Uterine Contractions", 0.0, 0.05, 0.003, step=0.001)
        light_decelerations = st.slider("Light Decelerations", 0.0, 0.05, 0.003, step=0.001)
        severe_decelerations = st.slider("Severe Decelerations", 0.0, 0.05, 0.0, step=0.001)
        prolongued_decelerations = st.slider("Prolongued Decelerations", 0.0, 0.05, 0.0, step=0.001)

    with st.expander("📊 Variability", expanded=True):
        abnormal_short_term_variability = st.slider("Abnormal Short Term Variability", 0, 100, 73)
        mean_value_of_short_term_variability = st.slider("Mean Short Term Variability", 0.0, 5.0, 0.5, step=0.1)
        percentage_of_time_with_abnormal_long_term_variability = st.slider("% Time Abnormal Long Term Variability", 0, 100, 43)
        mean_value_of_long_term_variability = st.slider("Mean Long Term Variability", 0.0, 20.0, 2.4, step=0.1)

# 6. Histogram values — fixed dataset averages
histogram_width = 70.0
histogram_min = 60.0
histogram_max = 160.0
histogram_number_of_peaks = 2.0
histogram_number_of_zeroes = 0.0
histogram_tendency = 0.0
histogram_variance = 70.0

# 7. Initialize session state
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
if "input_values" not in st.session_state:
    st.session_state.input_values = None

# 8. Predict button
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

    input_values = {
        "baseline": baseline,
        "accelerations": accelerations,
        "fetal_movement": fetal_movement,
        "uterine_contractions": uterine_contractions,
        "light_decelerations": light_decelerations,
        "severe_decelerations": severe_decelerations,
        "prolongued_decelerations": prolongued_decelerations,
        "abnormal_stv": abnormal_short_term_variability,
        "pct_abnormal_ltv": percentage_of_time_with_abnormal_long_term_variability,
        "mean_ltv": mean_value_of_long_term_variability
    }

    with st.spinner("Generating explanation..."):
        explanation = explain_prediction(int(prediction), confidence, input_values)

    st.session_state.prediction = class_map[int(prediction)]
    st.session_state.confidence = confidence
    st.session_state.explanation = explanation
    st.session_state.prediction_done = True
    st.session_state.history = []
    st.session_state.last_answer = ""
    st.session_state.input_values = input_values
    st.session_state.prediction_context = f"""
    Previous CTG Analysis Result:
    - Prediction: {class_map[int(prediction)]}
    - Confidence — Normal: {confidence[0]*100:.1f}%, Suspect: {confidence[1]*100:.1f}%, Pathological: {confidence[2]*100:.1f}%
    - Input Values: {input_values}
    - LLM Explanation: {explanation}
    """

# 9. Show prediction results
if st.session_state.prediction_done:
    st.subheader(f"Prediction: {st.session_state.prediction}")
    st.write(
        f"Confidence — Normal: {st.session_state.confidence[0]*100:.1f}% | "
        f"Suspect: {st.session_state.confidence[1]*100:.1f}% | "
        f"Pathological: {st.session_state.confidence[2]*100:.1f}%"
    )

    # Feature status chart
    st.plotly_chart(
        plot_feature_status(st.session_state.input_values),
        use_container_width=True
    )

    # Short LLM summary
    st.info(st.session_state.explanation)

    # 10. Follow-up chat
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

# 11. Disclaimer
st.markdown("---")
st.caption("⚠️ This tool is for clinical assistance only and does not replace professional medical diagnosis.")