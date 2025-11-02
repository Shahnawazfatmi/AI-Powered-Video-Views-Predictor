# 1️⃣ Imports
import streamlit as st
import joblib
from scipy import sparse
import numpy as np
from datetime import datetime

# 2️⃣ Page configuration
st.set_page_config(
    page_title="AI-Powered Video Views Predictor",
    page_icon="🤖",
    layout="wide"
)

# 3️⃣ Load saved objects
ensemble_model = joblib.load("stack_ensemble_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# 4️⃣ Prediction function
def predict_views_streamlit(title, description, publishedAt):
    try:
        publish_datetime = datetime.strptime(publishedAt, "%Y-%m-%dT%H:%M:%SZ")
    except:
        st.error("❌ Invalid date format! Use YYYY-MM-DDTHH:MM:SSZ")
        return None

    hour = publish_datetime.hour
    weekday = publish_datetime.weekday()
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    text = f"{title} {description}"
    text_features_new = tfidf.transform([text])
    numeric_features_new = sparse.csr_matrix([[hour_sin, hour_cos, weekday_sin, weekday_cos]])
    X_new = sparse.hstack([text_features_new, numeric_features_new])

    y_pred = ensemble_model.predict(X_new)
    y_pred_views = np.expm1(y_pred)
    return int(y_pred_views[0])

# 5️⃣ Custom CSS — AI aesthetic with gradient glow & animation
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

    .stApp {
        background: radial-gradient(circle at top left, #020617, #030a1a, #000);
        color: #e5e5e5;
        font-family: 'Poppins', sans-serif;
        overflow-x: hidden;
    }
    .title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        background: linear-gradient(90deg, #00b4d8, #7b2ff7, #f107a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 25px;
        letter-spacing: 1px;
        animation: glow 3s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 15px rgba(0,180,216,0.4); }
        to { text-shadow: 0 0 35px rgba(241,7,163,0.6); }
    }
    .subtitle {
        text-align: center;
        color: #b5b5b5;
        font-size: 18px;
        margin-bottom: 35px;
    }
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00b4d8, #7b2ff7, #f107a3, transparent);
        margin: 30px 0;
    }
    .stTextInput>div>div>input, textarea {
        background: linear-gradient(145deg, #0b0b0b, #111);
        color: #fafafa;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 12px;
        transition: 0.3s ease;
    }
    .stTextInput>div>div>input:focus, textarea:focus {
        border-color: #7b2ff7;
        box-shadow: 0 0 10px rgba(123,47,247,0.5);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00b4d8, #7b2ff7, #f107a3);
        background-size: 200% 200%;
        color: white;
        font-weight: 600;
        font-size: 17px;
        border: none;
        border-radius: 12px;
        height: 55px;
        width: 100%;
        transition: all 0.4s ease;
        box-shadow: 0 0 20px rgba(123,47,247,0.3);
    }
    .stButton>button:hover {
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 0 35px rgba(241,7,163,0.4);
    }
    .card {
        background: linear-gradient(135deg, rgba(17,17,17,0.8), rgba(27,27,27,0.4));
        backdrop-filter: blur(10px);
        padding: 35px;
        border-radius: 18px;
        margin-top: 40px;
        text-align: center;
        border: 1px solid rgba(0,180,216,0.2);
        box-shadow: 0 0 25px rgba(0,180,216,0.15);
        transition: all 0.4s ease;
        opacity: 0;
        animation: fadeIn 1s ease-in-out forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .card:hover {
        box-shadow: 0 0 50px rgba(241,7,163,0.35);
        border-color: rgba(241,7,163,0.5);
    }
    h2 {
        color: #00b4d8;
        font-size: 34px;
        margin-bottom: 12px;
    }
    p {
        color: #c9c9c9;
        font-size: 15px;
        margin-top: 5px;
    }
    footer {
        text-align: center;
        font-size: 13px;
        color: #777;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# 6️⃣ Header
st.markdown('<div class="title">AI-Powered Video Views Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict audience reach using next-gen machine learning insights 📈</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# 7️⃣ Input Section
with st.container():
    st.subheader("🎯 Enter Video Details")
    title = st.text_input("Video Title", placeholder="e.g. The Secret Formula for Viral Shorts")
    description = st.text_area("Video Description", placeholder="Briefly describe your video content...")
    publishedAt = st.text_input(
        "Publish Date & Time (YYYY-MM-DDTHH:MM:SSZ)",
        placeholder="2025-10-28T15:45:00Z"
    )

# 8️⃣ Prediction button & output
if st.button("🚀 Predict AI Estimated Views"):
    if not title.strip() or not description.strip() or not publishedAt.strip():
        st.warning("⚠️ Please fill all fields before prediction.")
    else:
        predicted_views = predict_views_streamlit(title, description, publishedAt)
        if predicted_views is not None:
            st.markdown(f"""
                <div class="card">
                    <h2>✨ Estimated Reach: {predicted_views:,} views</h2>
                    <p>Powered by VisionAI — transforming creator insights with machine learning precision.</p>
                </div>
            """, unsafe_allow_html=True)

# 9️⃣ Footer
st.markdown("""
    <footer>
        © 2025 VisionAI Labs · Turning data into creative intelligence 🚀
    </footer>
""", unsafe_allow_html=True)
