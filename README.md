# 🏥 AI-Powered Fetal Health Assistant

> Combining Machine Learning and AI to help healthcare professionals assess fetal health risk — and understand *why*.

🔗 **Live App:** [Click here to try it](https://your-app-link.streamlit.app) ← *update after deploying*

---

## 📌 Project Overview

Cardiotocograms (CTGs) are a cost-accessible tool used to monitor fetal health during pregnancy. This project builds an end-to-end AI system that:

1. Classifies fetal health as **Normal**, **Suspect**, or **Pathological** using a trained ML model
2. Uses an **LLM** to explain the prediction in plain English
3. Allows doctors or patients to **ask follow-up questions** about the diagnosis
4. Supports **CSV file upload** for automatic value reading or manual input

This project contributes to the United Nations' Sustainable Development Goal of reducing child and maternal mortality.

---

## 📊 Dataset

- **Source:** [Kaggle — Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
- **Records:** 2,126 CTG exam records
- **Features:** Fetal heart rate, accelerations, uterine contractions, and more
- **Target Classes:**
  - `1` → Normal
  - `2` → Suspect
  - `3` → Pathological

---

## 🧠 ML Model

- **Algorithm:** Random Forest Classifier
- **Class balancing:** `class_weight='balanced'`
- **Preprocessing:** StandardScaler

### Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Normal | 0.95 | 0.98 | 0.97 |
| Suspect | 0.91 | 0.78 | 0.84 |
| Pathological | 1.00 | 0.93 | 0.96 |
| **Overall Accuracy** | | | **0.95** |

---

## 🤖 AI (LLM) Layer

- Takes the ML prediction + confidence scores as input
- Uses **Groq API (LLaMA 3.3)** to generate a plain English explanation
- Maintains **chat history** for contextual follow-up questions
- System prompt tailored specifically for CTG and fetal health context

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | pandas, numpy, scikit-learn, joblib |
| Explainability | Feature importance |
| LLM | Groq API (LLaMA 3.3-70b) |
| UI | Streamlit |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/preetha-pallavi/fetal-health-assistant.git
cd fetal-health-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API key
Create a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🌿 Branches

| Branch | Description |
|---|---|
| `main` | ML model, EDA, core classification + full app |
| `llm-integration` | LLM development branch |

---

## 📁 Project Structure

```
fetal_health_project/
│
├── fetal_health_model.pkl             # Trained Random Forest model
├── fetal_scaler.pkl                   # Fitted StandardScaler
├── Fetal_Health_Classification.ipynb  # EDA + Model training notebook
├── Fetal_health_LLM.py                # Groq LLM logic + follow-up Q&A
├── app.py                             # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 💡 Key Features

- ✅ 95% overall classification accuracy
- ✅ Handles class imbalance
- ✅ Feature importance visualization
- ✅ CSV file upload for automatic value reading
- ✅ LLM-powered plain English explanations
- ✅ Interactive follow-up Q&A with chat history
- ✅ Medical disclaimer included

---

## 🌍 Why This Matters

> 295,000 maternal deaths occur annually during pregnancy and childbirth. 94% happen in low-resource settings — most preventable.

This tool aims to make fetal health assessment more accessible and interpretable for healthcare professionals worldwide.

---

## 👤 Author

**Preetha Pallavi**
- GitHub: [@preetha-pallavi](https://github.com/preetha-pallavi)
- LinkedIn: [PreethaPallavi](https://www.linkedin.com/in/preetha-pallavi/)