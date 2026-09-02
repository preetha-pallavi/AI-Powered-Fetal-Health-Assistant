from openai import OpenAI
from dotenv import load_dotenv
import os

# 1. Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# System prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a specialist in fetal health and Cardiotocography (CTG).
    You help OBGYNs and midwives interpret CTG results.
    Always relate your answers specifically to fetal heart rate,
    uterine contractions, fetal movement, and CTG patterns.
    Never give generic medical advice — always tie answers back to CTG context.
    Always explain in simple, everyday language that a non-medical person can understand.
    Avoid medical jargon — if you must use a medical term, explain it in brackets.
    Always respond in short paragraphs, never in bullet points or numbered lists."""
}

# 2. Explain prediction
def explain_prediction(prediction, confidence, input_values):
    class_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    label = class_map[prediction]

    prompt = f"""
    The CTG analysis ML model predicted: {label}
    Confidence scores — Normal: {confidence[0]*100:.1f}%, Suspect: {confidence[1]*100:.1f}%, Pathological: {confidence[2]*100:.1f}%

    The actual CTG input values were:
    - Baseline FHR: {input_values['baseline']}
    - Accelerations: {input_values['accelerations']}
    - Fetal Movement: {input_values['fetal_movement']}
    - Uterine Contractions: {input_values['uterine_contractions']}
    - Light Decelerations: {input_values['light_decelerations']}
    - Severe Decelerations: {input_values['severe_decelerations']}
    - Prolongued Decelerations: {input_values['prolongued_decelerations']}
    - Abnormal Short Term Variability: {input_values['abnormal_stv']}
    - % Time Abnormal Long Term Variability: {input_values['pct_abnormal_ltv']}
    - Mean Long Term Variability: {input_values['mean_ltv']}

    Based on these values, write 1-2 sentences only:
    - Summarise why the model predicted {label} based on the most concerning values
    - Do not give clinical advice
    - Do not mention cesarean or medical procedures
    - Keep it simple, non-alarming and in everyday language
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[SYSTEM_PROMPT, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 3. Follow-up questions
def ask_followup(question, history):
    messages = [SYSTEM_PROMPT] + history + [{"role": "user", "content": question + "\n\nPlease explain in simple everyday language, avoid medical jargon."}]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return answer, history