import streamlit as st
import pandas as pd
import random
import datetime
from transformers import pipeline
import cv2
import numpy as np

# --- Initialize AI model ---
try:
    generator = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    generator = None
    st.warning("⚠️ AI text generator not available, using backup quotes.")

# --- Page setup ---
st.set_page_config(page_title="Mood Companion", page_icon="🌤️")
st.title("🌤️ Mood Companion (AI + Mood Tracker)")
st.write("Type your mood, take a selfie, and receive a kind message 💛")

# --- Mood input section ---
mood = st.selectbox(
    "💭 How are you feeling today?",
    ["😐 Neutral", "😞 Sad", "😡 Angry", "😰 Stressed", "😊 Happy", "😴 Tired"]
)

# --- Optional user text input ---
text_input = st.text_input("You can also describe your mood in your own words:")

if text_input.strip() != "":
    mood = text_input

# --- AI response ---
quotes = [
    "You are enough. Just as you are. 💛",
    "Progress, not perfection.",
    "You’re growing through what you’re going through.",
    "Take small steps forward — they still count.",
    "Even slow healing is healing. 🌱",
    "You’ve survived 100% of your bad days."
]

if st.button("💬 Show Message"):
    if generator:
        prompt = f"The user feels {mood}. Write one short gentle motivational sentence:"
        ai_reply = generator(prompt, max_length=40, num_return_sequences=1)[0]["generated_text"]
    else:
        ai_reply = random.choice(quotes)
    st.success(ai_reply)
    st.info(random.choice(quotes))

# --- Mood Tracker ---
today = datetime.date.today()
try:
    df = pd.read_csv("mood_log.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=["Date", "Mood"])

if st.button("📘 Save my mood"):
    new_entry = pd.DataFrame({"Date": [today], "Mood": [mood]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.drop_duplicates(subset="Date", keep="last", inplace=True)
    df.to_csv("mood_log.csv", index=False)
    st.success("Mood saved successfully! 💖")

if not df.empty:
    st.write("### 📊 Your Recent Mood History")
    st.dataframe(df.tail(7))

# --- Camera section (Cloud Safe) ---
st.write("### 📸 Take a Selfie (Optional)")
img = st.camera_input("Capture your current mood:")

if img is not None:
    st.image(img, caption="Nice photo! 😊", use_column_width=True)
    # Convert to OpenCV image
    bytes_data = img.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    st.success("Image captured successfully!")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Mood Companion © 2025")
