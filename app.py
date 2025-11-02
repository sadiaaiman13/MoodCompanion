import streamlit as st
import pandas as pd
import random
import datetime
import sys

# --- Try to import optional libraries safely ---
try:
    from transformers import pipeline
    generator = pipeline("text-generation", model="distilgpt2")
except Exception:
    generator = None

try:
    import cv2
    cv2_available = True
except ImportError:
    cv2_available = False

try:
    import speech_recognition as sr
    voice_available = True
except ImportError:
    voice_available = False


# --- Streamlit Page Setup ---
st.set_page_config(page_title="🌤️ Mood Companion", page_icon="🌤️")
st.title("🌤️ Mood Companion (AI + Tracker)")
st.write("Track your mood, get kind AI messages, and see your progress 💛")

# --- Mood Input ---
st.write("🎙️ You can type your mood or choose one below:")
mood = st.selectbox(
    "Select your current mood:",
    ["😐 Neutral", "😞 Sad", "😡 Angry", "😰 Stressed", "😊 Happy", "😴 Tired"]
)

# --- Voice Input (Local Only) ---
if voice_available and not st.secrets.get("on_cloud", False):
    if st.button("🎧 Record my voice"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... speak your mood (e.g., 'sad', 'happy')")
            audio = r.listen(source, phrase_time_limit=4)
        try:
            text = r.recognize_google(audio).lower()
            st.success(f"You said: {text}")
            if "sad" in text: mood = "😞 Sad"
            elif "stress" in text: mood = "😰 Stressed"
            elif "angry" in text: mood = "😡 Angry"
            elif "tired" in text: mood = "😴 Tired"
            elif "happy" in text: mood = "😊 Happy"
            else: mood = "😐 Neutral"
        except Exception:
            st.error("Sorry, I couldn’t understand you.")
else:
    st.info("🎤 Voice recording not available here (works on local only).")

# --- Quotes ---
quotes = [
    "You are enough. Just as you are. 💛",
    "Progress, not perfection.",
    "You’re growing through what you’re going through.",
    "Take small steps forward — they still count.",
    "Even slow healing is healing. 🌱",
    "You’ve survived 100% of your bad days."
]

# --- AI Response ---
if st.button("💬 Show Message"):
    if generator:
        prompt = f"The user feels {mood}. Write one short gentle motivational sentence:"
        ai_reply = generator(prompt, max_length=40, num_return_sequences=1)[0]["generated_text"]
        st.success(ai_reply)
    else:
        st.success(random.choice(quotes))
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

# --- Optional Local Camera (Skip on Cloud) ---
if cv2_available and not st.secrets.get("on_cloud", False):
    st.write("### 🎥 Detect Emotion from Camera (Smile Detector)")
    if st.button("Open Camera"):
        st.info("Press 'q' to close camera window.")
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
                roi_gray = gray[y:y+h, x:x+w]
                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                label = "😊 Happy" if len(smiles) > 0 else "😐 Neutral"
                color = (0, 255, 0) if len(smiles) > 0 else (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("Mood Companion - Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        st.success("Camera closed 💛")
else:
    st.info("📷 Camera detection available only on local setup.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Mood Companion © 2025")
