"""
Streamlit Frontend for PEFT Fake News Detector

A simple, beautiful web interface for analyzing news articles using the 
PEFT-adapted RoBERTa model with LoRA.
"""

import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalistic styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextArea textarea {
        font-size: 15px;
        border-radius: 8px;
    }
    .stButton > button {
        width: 100%;
    }
    .prediction-box {
        padding: 16px 20px;
        border-radius: 8px;
        margin: 20px 0;
        border: 1px solid;
    }
    .fake-news {
        background-color: #ffebee;
        border-color: #f44336;
    }
    .true-news {
        background-color: #e8f5e9;
        border-color: #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration (using port 5001 to avoid macOS AirPlay on 5000)
API_URL = "http://localhost:5001/api/predict"
HEALTH_URL = "http://localhost:5001/api/health"

def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get(HEALTH_URL, timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_news(text):
    """Send text to backend for prediction."""
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to backend. Please ensure the Python backend is running on port 5000."
    except Exception as e:
        return None, f"Error: {str(e)}"

# Sample examples - Only examples that model predicts correctly
TRUE_NEWS = {
    "Federal Reserve": "The Federal Reserve announced today that it will maintain current interest rates at their existing levels. The decision was made following a review of economic indicators including inflation rates and employment figures.",
    "Climate Summit": "World leaders at the UN Climate Summit reached a historic agreement to reduce carbon emissions by 50% by 2030. The accord was signed by 195 countries.",
    "Trade Agreement": "The United States and European Union signed a new trade agreement reducing tariffs on agricultural products. The deal is expected to boost exports for both regions.",
    "International Relations": "The United Nations General Assembly convened today to discuss global peacekeeping efforts. Representatives from 193 member nations are participating.",
    "Energy Sector": "Oil prices declined 2 percent today amid concerns about global demand. Crude oil futures settled at 78 dollars per barrel."
}

FAKE_NEWS = {
    "Chocolate Planet": "Breaking news: Scientists discover a new planet made entirely of chocolate. NASA confirms the sweet discovery through advanced telescope observations.",
    "Miracle Pill": "Doctors stunned by new pill that makes you lose 50 pounds overnight without any exercise or diet changes. Big pharma trying to hide this miracle cure!",
    "Celebrity Plot": "SHOCKING: Famous celebrities caught in secret underground meeting planning to control the world economy. Leaked documents reveal their sinister plan.",
    "Ancient Aliens": "Scientists finally admit that aliens built the Egyptian pyramids after discovering advanced technology inside. Government covered this up for decades!",
    "Free Money": "The government is giving away $10,000 to every citizen but doesn't want you to know. Click here to claim your free money before program ends!"
}

# Header
st.title("🔍 Fake News Detector")
st.caption("AI-powered verification using RoBERTa + LoRA")
st.divider()

# Example dropdowns
# Initialize dropdown state
if 'selected_true' not in st.session_state:
    st.session_state['selected_true'] = 0
if 'selected_fake' not in st.session_state:
    st.session_state['selected_fake'] = 0

col1, col2 = st.columns(2)
with col1:
    true_choice = st.selectbox(
        "📰 True News Examples", 
        ["Select..."] + list(TRUE_NEWS.keys()),
        index=st.session_state['selected_true']
    )
    if true_choice != "Select...":
        st.session_state['example_text'] = TRUE_NEWS[true_choice]
        st.session_state['selected_fake'] = 0  # Reset fake dropdown
        st.session_state['selected_true'] = (["Select..."] + list(TRUE_NEWS.keys())).index(true_choice)

with col2:
    fake_choice = st.selectbox(
        "⚠️ Fake News Examples", 
        ["Select..."] + list(FAKE_NEWS.keys()),
        index=st.session_state['selected_fake']
    )
    if fake_choice != "Select...":
        st.session_state['example_text'] = FAKE_NEWS[fake_choice]
        st.session_state['selected_true'] = 0  # Reset true dropdown
        st.session_state['selected_fake'] = (["Select..."] + list(FAKE_NEWS.keys())).index(fake_choice)

st.divider()

# Text input
default_text = st.session_state.get('example_text', '')
news_text = st.text_area(
    "Enter News Article",
    value=default_text,
    height=150,
    placeholder="Paste a news article here..."
)

# Buttons - same size
col1, col2 = st.columns(2)
with col1:
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state['example_text'] = ''
        st.rerun()

# Analysis results
if analyze_btn:
    if not news_text or len(news_text.strip()) < 50:
        st.warning("Please enter at least 50 characters.")
    else:
        with st.spinner("Analyzing..."):
            result, error = predict_news(news_text)
            
            if error:
                st.error(error)
            elif result:
                st.divider()
                
                # Prediction
                prediction = result['prediction']
                confidence = result['confidence']
                
                if prediction == "Fake News":
                    st.markdown(f"""
                    <div class="prediction-box fake-news">
                        <h2 style="color: #000000; margin: 0;">❌ Fake News</h2>
                        <p style="font-size: 18px; margin: 10px 0 0 0; color: #000000;">Confidence: {confidence}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box true-news">
                        <h2 style="color: #388e3c; margin: 0;">✅ True News</h2>
                        <p style="font-size: 18px; margin: 10px 0 0 0;">Confidence: {confidence}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.subheader("Probability Distribution")
                col_true, col_fake = st.columns(2)
                
                with col_true:
                    st.metric("True News", f"{result['true_probability']}%")
                    st.progress(result['true_probability'] / 100)
                
                with col_fake:
                    st.metric("Fake News", f"{result['fake_probability']}%")
                    st.progress(result['fake_probability'] / 100)

# Footer
st.divider()
st.caption("Built with RoBERTa + LoRA • PEFT Fine-Tuning • UmdTask96 Fall 2025")
