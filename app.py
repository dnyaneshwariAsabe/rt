import streamlit as st
import pickle
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ML Model Predictor",
    page_icon="🤖",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
        animation: fadeIn 2s ease-in;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #f0f0f0;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #00c9a7;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #008f7a;
        transform: scale(1.05);
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="title">🤖 ML Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload inputs and get instant predictions</div>', unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    with open("model (5).pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ------------------ INPUT SECTION ------------------
st.subheader("📥 Enter Input Features")

# ⚠️ CHANGE number of inputs based on your model
# Example: 3 features
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# ------------------ PREDICT BUTTON ------------------
if st.button("🚀 Predict"):
    try:
        input_data = np.array([[feature1, feature2, feature3]])
        prediction = model.predict(input_data)

        st.success(f"🎯 Prediction: {prediction[0]}")

        # 🎉 Animation (balloons)
        st.balloons()

    except Exception as e:
        st.error(f"Error: {e}")

# ------------------ FOOTER ------------------
st.markdown("""
    <hr>
    <center>Made with ❤️ using Streamlit</center>
""", unsafe_allow_html=True)
