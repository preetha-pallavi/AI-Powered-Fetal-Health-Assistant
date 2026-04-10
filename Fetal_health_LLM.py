from groq import Groq
from dotenv import load_dotenv
import os

# 1. Load API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System prompt — gives LLM its role and context
SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a specialist in fetal health and Cardiotocography (CTG).
    You help OBGYNs and midwives interpret CTG results.
    Always relate your answers specifically to fetal heart rate,
    uterine contractions, fetal movement, and CTG patterns.
    Never give generic medical advice — always tie answers back to CTG context.
    Be concise, clear and professional. Prefererably in points
    """
}

# 2. Explain prediction after model classifies
def explain_prediction(prediction, confidence):
    class_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    label = class_map[prediction]

    prompt = f"""
    The CTG analysis ML model predicted: {label}
    Confidence scores — Normal: {confidence[0]:.2f}, Suspect: {confidence[1]:.2f}, Pathological: {confidence[2]:.2f}
    
    Explain this CTG result to a healthcare professional.
    Mention what this means for fetal heart rate patterns and what clinical action should be taken.
    Keep it under 5 sentences and to the point . Prefererably in points
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[SYSTEM_PROMPT, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 3. Follow-up questions (keeps chat history)
def ask_followup(question, history):
    # Always inject system prompt at the start
    messages = [SYSTEM_PROMPT] + history + [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    answer = response.choices[0].message.content
    # Only store user/assistant in history, not sys_tem
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return answer, history