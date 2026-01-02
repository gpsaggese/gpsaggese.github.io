import streamlit as st
import pandas as pd
import torch
import plotly.express as px
import os
import csv
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from model_training import utils
from sklearn.metrics import confusion_matrix

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Airline Sentiment Lab", layout="wide")
device = utils.get_device()

# File to store active learning feedback
FEEDBACK_FILE = "./data/dpo_feedback.csv"

# ==========================================
# PAGE SETUP
# ==========================================
st.title("✈️ Airline Sentiment Analysis Lab")
st.markdown("Compare Baseline BERT vs GPT-2 Baseline vs GPT-2 (SFT) vs GPT-2 (DPO + Active Learning)")

page = st.sidebar.radio("Navigate", ["Live Playground", "Model Evaluation"])

# ==========================================
# HELPER: LOAD MODELS
# ==========================================
MODELS = {
    "Baseline BERT": {"path": "blank4hd/airline-sentiment-bert-baseline", "type": "bert"},
    "Baseline GPT-2": {"path": "blank4hd/airline-sentiment-baseline-gpt2-sft", "type": "gpt2"},
    "Improved SFT":  {"path": "blank4hd/airline-sentiment-gpt2-improved-sft", "type": "gpt2"},
    "DPO (Active)":  {"path": "blank4hd/airline-sentiment-gpt2-dpo-active-learning", "type": "gpt2"},
}

@st.cache_resource
def load_model(repo_id, model_type):
    try:
        if model_type == "bert":
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForSequenceClassification.from_pretrained(repo_id).to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(repo_id).to(device)
            model.config.pad_token_id = tokenizer.eos_token_id
        return tokenizer, model
    except Exception as e:
        return None, None

def save_feedback(text, user_label, rejected_label):
    """Saves user feedback as a preference pair for DPO training"""
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "text", "chosen_label", "rejected_label"])
        writer.writerow([datetime.now(), text, user_label, rejected_label])

# ==========================================
# PAGE 1: LIVE PLAYGROUND + ACTIVE LEARNING
# ==========================================
if page == "Live Playground":
    st.header("🧪 Live Playground")
    
    # 1. Input Section
    col_input, col_controls = st.columns([3, 1])
    with col_input:
        user_text = st.text_area("Enter a Tweet:", "The flight was okay but the food was cold.", height=100)
    with col_controls:
        selected_models = st.multiselect("Select Models", list(MODELS.keys()), default=list(MODELS.keys()))
        run_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    # State management for predictions
    if "last_preds" not in st.session_state:
        st.session_state.last_preds = {}

    if run_btn:
        if not user_text.strip():
            st.warning("Please enter text first.")
        else:
            st.session_state.last_preds = {} # Reset
            cols = st.columns(len(selected_models))
            
            for idx, name in enumerate(selected_models):
                config = MODELS[name]
                with cols[idx]:
                    st.subheader(name)
                    with st.spinner("Thinking..."):
                        tokenizer, model = load_model(config["path"], config["type"])
                    
                    if model:
                        try:
                            # INFERENCE LOGIC
                            if config["type"] == "bert":
                                inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=64).to(device)
                                with torch.no_grad():
                                    probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
                                    pred = ["negative", "neutral", "positive"][torch.argmax(probs).item()]
                            else:
                                prompt = utils.build_gpt2_prompt(user_text)
                                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
                                with torch.no_grad():
                                    out = model.generate(**inputs, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id, do_sample=False)
                                pred = utils.extract_label(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
                            
                            # STORE PREDICTION
                            st.session_state.last_preds[name] = pred
                            
                            # DISPLAY
                            color = {"positive": "green", "negative": "red", "neutral": "gray"}.get(pred, "blue")
                            st.markdown(f"<h3 style='color: {color}; text-align: center; border: 2px solid {color}; border-radius: 10px; padding: 10px;'>{pred.upper()}</h3>", unsafe_allow_html=True)
                        
                        except Exception as e:
                            st.error(f"Error: {e}")

    # 2. ACTIVE LEARNING FEEDBACK LOOP
    if st.session_state.last_preds:
        st.divider()
        st.subheader("📝 Active Learning Loop (Human Feedback)")
        st.info("If the models got it wrong, teach them! Your feedback creates the preference pairs used to train the DPO model.")
        
        with st.form("feedback_form"):
            f_col1, f_col2 = st.columns(2)
            
            with f_col1:
                st.write("**Which sentiment is ACTUALLY correct?**")
                correct_label = st.radio("Correct Label:", ["negative", "neutral", "positive"], horizontal=True)
            
            with f_col2:
                st.write("**Which label should be rejected?**")
                # Default to whatever the DPO model predicted, or the first available prediction
                default_reject = st.session_state.last_preds.get("DPO (Active)", list(st.session_state.last_preds.values())[0])
                rejected_label = st.selectbox("Incorrect Prediction to Penalize:", ["negative", "neutral", "positive"], index=["negative", "neutral", "positive"].index(default_reject) if default_reject in ["negative", "neutral", "positive"] else 0)

            submit_feedback = st.form_submit_button("✅ Submit Feedback to DPO Dataset")
            
            if submit_feedback:
                if correct_label == rejected_label:
                    st.warning("The Chosen and Rejected labels cannot be the same.")
                else:
                    save_feedback(user_text, correct_label, rejected_label)
                    st.success(f"Feedback Saved! Pair created: (Chosen: {correct_label.upper()} > Rejected: {rejected_label.upper()})")
                    st.balloons()
                    
        # Show recent feedback
        if os.path.exists(FEEDBACK_FILE):
            with st.expander("View Recent Feedback Data"):
                st.dataframe(pd.read_csv(FEEDBACK_FILE).tail(5))

# ==========================================
# PAGE 2: MODEL EVALUATION
# ==========================================
elif page == "Model Evaluation":
    st.header("📊 Model Performance Analysis")
    
    try:
        # Load Data
        df_metrics = pd.read_csv("./data/evaluation_results.csv")
        df_preds = pd.read_csv("./data/predictions.csv")
        
        # TABS
        tab1, tab2, tab3 = st.tabs(["📈 Metrics Leaderboard", "🎨 Confusion Matrix", "🕵️ Error Explorer"])
        
        # --- TAB 1: METRICS ---
        with tab1:
            st.subheader("Accuracy & F1 Scores")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df_metrics.style.highlight_max(axis=0, color="#d4edda"), width="stretch")
            with col2:
                fig = px.bar(df_metrics, x="Model", y=["Accuracy", "Macro F1"], barmode="group", title="Performance Comparison")
                st.plotly_chart(fig, width="stretch")

        # --- TAB 2: CONFUSION MATRIX ---
        with tab2:
            st.subheader("Confusion Matrix Heatmap")
            model_cm = st.selectbox("Select Model to Visualize", df_preds.columns[2:]) # Skip text/truth cols
            
            labels = ["negative", "neutral", "positive"]
            cm = confusion_matrix(df_preds["airline_sentiment"], df_preds[model_cm], labels=labels)
            
            fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, 
                               labels=dict(x="Predicted", y="True Label", color="Count"),
                               color_continuous_scale="Blues")
            st.plotly_chart(fig_cm)
            
        # --- TAB 3: ERROR EXPLORER ---
        with tab3:
            st.subheader("Hall of Shame (Where Models Failed)")
            
            # Filter options
            st.write("Show tweets where:")
            col_fail, col_succ = st.columns(2)
            with col_fail:
                fail_model = st.selectbox("This Model FAILED:", ["Any"] + list(df_preds.columns[2:]))
            with col_succ:
                succ_model = st.selectbox("But this Model SUCCEEDED (Optional):", ["None"] + list(df_preds.columns[2:]))
            
            # Filtering Logic
            filtered_df = df_preds.copy()
            
            if fail_model != "Any":
                filtered_df = filtered_df[filtered_df[fail_model] != filtered_df["airline_sentiment"]]
            
            if succ_model != "None":
                filtered_df = filtered_df[filtered_df[succ_model] == filtered_df["airline_sentiment"]]
            
            st.write(f"Found **{len(filtered_df)}** tweets matching criteria.")
            st.dataframe(filtered_df, width="stretch")

    except FileNotFoundError:
        st.error("⚠️ Results files not found!")
        st.info("Please run `python compare_models.py` first to generate 'predictions.csv' and 'evaluation_results.csv'.")