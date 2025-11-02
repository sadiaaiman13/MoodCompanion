import streamlit as st
import pandas as pd
import random
import datetime
try:
    import speech_recognition as sr
    voice_available = True
except ImportError:
    voice_available = False

from transformers import pipeline
import cv2


# --- AI text generator ---
generator = pipeline("text-generation", model="distilgpt2")

st.set_page_config(page_title="Mood Companion", page_icon="🌤️")

st.title("🌤️ Mood Companion (Edge AI + Voice + Tracker)")
st.write("Speak, smile, or share your mood — I’ll listen and respond gently 💛")

# --- Mood input section ---
st.write("🎙️ You can type or say your mood aloud")

mood = st.selectbox(
    "Or choose your current mood:",
    ["😐 Neutral", "😞 Sad", "😡 Angry", "😰 Stressed", "😊 Happy", "😴 Tired"]
)

# --- Voice input ---
st.write("🎙️ You can type your mood — or use your voice if available")

if voice_available:
    if st.button("🎧 Record my voice"):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("Listening... speak your mood (e.g., 'sad', 'happy')")
                audio = r.listen(source, phrase_time_limit=4)
            text = r.recognize_google(audio).lower()
            st.success(f"You said: {text}")
            if "sad" in text: mood = "😞 Sad"
            elif "stress" in text: mood = "😰 Stressed"
            elif "angry" in text: mood = "😡 Angry"
            elif "tired" in text: mood = "😴 Tired"
            elif "happy" in text: mood = "😊 Happy"
            else: mood = "😐 Neutral"
        except Exception as e:
            st.error("Sorry, I couldn’t process your voice input.")
else:
    st.info("🎤 Voice input not available on this platform — please select your mood manually.")


# --- Quotes for variety ---
quotes = [
    "You are enough. Just as you are. 💛",
    "Progress, not perfection.",
    "You’re growing through what you’re going through.",
    "Take small steps forward — they still count.",
    "Even slow healing is healing. 🌱",
    "You’ve survived 100% of your bad days."
]

# --- AI reply ---
if st.button("💬 Show Message"):
    prompt = f"The user feels {mood}. Write one short gentle motivational sentence:"
    ai_reply = generator(prompt, max_length=40, num_return_sequences=1)[0]["generated_text"]
    st.success(ai_reply)
    st.info(random.choice(quotes))

# --- Mood Tracker ---
today = datetime.date.today()
try:
    df = pd.read_csv("mood_log.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=["Date", "Mood"])

if st.button("Save my mood 📘"):
    new_entry = pd.DataFrame({"Date": [today], "Mood": [mood]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.drop_duplicates(subset="Date", keep="last", inplace=True)
    df.to_csv("mood_log.csv", index=False)
    st.success("Mood saved successfully! 💖")

if not df.empty:
    st.write("### 📊 Your Recent Mood History")
    st.dataframe(df.tail(7))

# --- Edge AI Camera Emotion (NO Kaggle model needed) ---
st.write("### 🎥 Detect Emotion from Camera (Smile Detector)")

if st.button("Open Camera"):
    st.info("Press 'q' in the camera window to close it.")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            if len(smiles) > 0:
                label = "😊 Happy"
                color = (0, 255, 0)
            else:
                label = "😐 Neutral"
                color = (255, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Mood Companion - Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("Camera closed. Hope that smile felt nice 💛")
