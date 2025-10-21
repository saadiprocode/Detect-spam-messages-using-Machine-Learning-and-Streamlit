# app/streamlit_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.utils import load_model, load_vectorizer

# -------------------------------------
# 🌟 PAGE CONFIGURATION
# -------------------------------------
st.set_page_config(
    page_title='SMS Spam Detector',
    page_icon='📩',
    layout='centered'
)

# Load model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# -------------------------------------
# 🌗 DARK/LIGHT MODE TOGGLE
# -------------------------------------
st.sidebar.header("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme:", ["Light", "Dark"])

bg_color = "#ffffff" if theme == "Light" else "#0e1117"
text_color = "#000000" if theme == "Light" else "#fafafa"
box_bg = "rgba(240, 240, 255, 0.5)" if theme == "Light" else "rgba(30, 30, 40, 0.5)"

# -------------------------------------
# 💅 CUSTOM STYLING
# -------------------------------------
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .main-title {{
            text-align: center;
            font-size: 2.4em;
            font-weight: bold;
            margin-bottom: 0.4em;
            color: #4e9af1;
        }}
        .sub-title {{
            text-align: center;
            color: #888;
            font-size: 1.1em;
            margin-bottom: 1.5em;
        }}
        .card {{
            background: {box_bg};
            border-radius: 16px;
            padding: 1.8em;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            transition: 0.3s;
        }}
        .card:hover {{
            transform: scale(1.02);
        }}
        .result {{
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
            padding: 1.2em;
            border-radius: 12px;
            margin-top: 20px;
        }}
        .spam {{
            background: linear-gradient(135deg, #ffb3b3, #ff6666);
            color: #400000;
        }}
        .ham {{
            background: linear-gradient(135deg, #b3ffd9, #00cc66);
            color: #002b0e;
        }}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------
# 🧠 HEADER
# -------------------------------------
st.markdown('<div class="main-title">📩 SMS Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Detect spam messages using a trained Machine Learning model ⚡</div>', unsafe_allow_html=True)

# -------------------------------------
# 💬 SAMPLE MESSAGES
# -------------------------------------
sample_messages = {
    "Select a sample message": "",
    "💰 Win a free iPhone! Click the link to claim your prize now!": "",
    "👋 Hey! Are we still meeting tomorrow for lunch?": "",
    "🎯 You have been selected for a $500 Amazon voucher. Reply YES to claim!": "",
    "🏦 Your account has been temporarily locked. Verify your details at this link.": "",
    "✨ Let's catch up soon! Been so long since we talked.": ""
}

st.markdown('<div class="card">', unsafe_allow_html=True)
selected = st.selectbox("📱 Try a sample message:", list(sample_messages.keys()))
text = st.text_area("✏️ Or enter your own message:", value=selected if selected != "Select a sample message" else "", height=130)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------
# 🔍 PREDICTION
# -------------------------------------
if st.button("🔍 Analyze Message"):
    if not text.strip():
        st.warning("⚠️ Please type or select a message first.")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        
        if pred == 1:
            st.markdown('<div class="result spam">🚨 This message is likely SPAM!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result ham">✅ This message seems SAFE (Not Spam)</div>', unsafe_allow_html=True)

# -------------------------------------
# 🪶 FOOTER
# -------------------------------------
st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:{text_color};'>Made with ❤️ using <b>Streamlit</b><br>by <b>Muhammad Saad</b></div>",
    unsafe_allow_html=True
)
