"""
Fresh vs Rotten Fruit Detection - Streamlit web app.

Run with:  streamlit run app.py

All ML logic lives in utils/prediction.py. This file only handles the UI.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from utils.prediction import (
    ALLOWED_EXTENSIONS,
    load_image,
    load_trained_model,
    predict_freshness,
    preprocess_image
)

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fresh vs Rotten Fruit Detection",
    page_icon="🍎",
    layout="wide",
)

# Metrics measured on the held-out test set (NOT a per-image guarantee)
METRICS = {
    "Training Accuracy": "97.82%",
    "Validation Accuracy": "98.35%",
    "Test Accuracy": "98.15%",
    "Test Loss": "0.0490",
}

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white; padding: 2.2rem 2.4rem; border-radius: 18px; margin-bottom: 1.5rem;
}
.hero h1 {color: white; margin: 0 0 .5rem 0; font-size: 2.2rem;}
.hero p  {color: #E0E7FF; margin: 0; font-size: 1.05rem;}
.card {
    background: white; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 1.2rem 1.4rem; box-shadow: 0 1px 3px rgba(15,23,42,.05);
    height: 100%;
}
.metric-card {text-align: center;}
.metric-card .value {font-size: 1.9rem; font-weight: 700; color: #4F46E5;}
.metric-card .label {color: #64748B; font-size: .85rem; text-transform: uppercase; letter-spacing: .05em;}
.result {border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; border-left: 8px solid;}
.result.fresh  {background: #ECFDF5; border-color: #10B981; color: #065F46;}
.result.rotten {background: #FEF2F2; border-color: #EF4444; color: #991B1B;}
.result h2 {margin: 0 0 .3rem 0; color: inherit;}
.result .conf {font-size: 1.05rem;}
.step {color: #475569; line-height: 1.6;}
.small-note {color: #64748B; font-size: .85rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



@st.cache_resource(show_spinner="Downloading and loading CNN model from Hugging Face...")
def get_model():
    """Download and load the trained model once."""
    return load_trained_model()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("history", [])       # list of dicts
    st.session_state.setdefault("uploader_key", 0)   # bump to reset the uploader
    st.session_state.setdefault("last_result", None)  # (file name, PredictionResult)


def reset_upload():
    """Clear the uploaded image and last result."""
    st.session_state.uploader_key += 1
    st.session_state.last_result = None


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
def metric_card(label: str, value: str) -> str:
    return (f'<div class="card metric-card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>')


def show_result(result):
    """Render the result card, probabilities and confidence bars."""
    is_fresh = result.label == "Fresh"
    css = "fresh" if is_fresh else "rotten"
    icon = "✅" if is_fresh else "⚠️"
    st.markdown(
        f'<div class="result {css}"><h2>{icon} {result.label} Fruit Detected</h2>'
        f'<div class="conf">Confidence: <b>{result.confidence:.2f}%</b></div></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("🟢 Probability of Fresh", f"{result.fresh_prob:.2f}%")
    c2.metric("🔴 Probability of Rotten", f"{result.rotten_prob:.2f}%")
    st.progress(min(result.fresh_prob / 100, 1.0), text="Fresh")
    st.progress(min(result.rotten_prob / 100, 1.0), text="Rotten")

    if result.confidence < 70:
        st.warning("Low confidence - the model is unsure about this image. "
                   "Try a clearer, well-lit photo of a single fruit.")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home():
    st.markdown(
        '<div class="hero"><h1>🍎 Fresh vs Rotten Fruit Detection</h1>'
        "<p>A custom Convolutional Neural Network that looks at a fruit photo "
        "and decides whether the fruit is fresh or rotten.</p></div>",
        unsafe_allow_html=True,
    )

    st.subheader("How it works")
    cols = st.columns(4)
    steps = [
        ("1. Upload", "Add a JPG, JPEG or PNG photo of a fruit."),
        ("2. Preprocess", "The image is converted to RGB, resized to 224×224 and scaled to 0-1."),
        ("3. CNN analysis", "Convolution layers extract textures, colours and spots."),
        ("4. Result", "A sigmoid output gives the probability of Rotten. ≥ 0.5 means Rotten."),
    ]
    for col, (title, text) in zip(cols, steps):
        col.markdown(f'<div class="card"><b>{title}</b><p class="step">{text}</p></div>',
                     unsafe_allow_html=True)

    st.write("")
    st.subheader("Model performance (test set)")
    cols = st.columns(4)
    for col, (label, value) in zip(cols, METRICS.items()):
        col.markdown(metric_card(label, value), unsafe_allow_html=True)
    st.markdown(
        '<p class="small-note">These are measured on the dataset\'s held-out test set. '
        "They are <b>not</b> a guarantee for every individual prediction, and real-world "
        "photos (different lighting, backgrounds or fruit types) can perform worse.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Getting started")
    st.info("Open **Fruit Detection** in the sidebar, upload a photo, and press "
            "**Analyze Fruit**. Best results: one fruit, good lighting, plain background.")


def page_detection():
    st.title("Fruit Detection")
    st.caption("Upload a fruit photo and let the CNN classify it as Fresh or Rotten.")

    # Load model (show a friendly error if missing)
    try:
        model = get_model()

    except Exception as exc:
        st.error(
            "Unable to download or load the trained model from Hugging Face."
        )
        st.exception(exc)
        st.stop()

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("1. Upload image")
        uploaded = st.file_uploader(
            "Drag and drop a fruit image, or click to browse (JPG, JPEG, PNG)",
            type=ALLOWED_EXTENSIONS,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        image = None
        if uploaded is not None:
            try:
                image = load_image(uploaded)
                st.image(image, caption=uploaded.name, use_container_width=True)
            except ValueError as err:
                st.error(str(err))

        b1, b2 = st.columns(2)
        analyze = b1.button("Analyze Fruit", type="primary",
                            use_container_width=True, disabled=image is None)
        b2.button("Reset / Upload another", on_click=reset_upload, use_container_width=True)

        if uploaded is None:
            st.markdown('<p class="small-note">Tip: use a single fruit, good lighting and a '
                        "plain background. This model was trained on apples, bananas and "
                        "oranges.</p>", unsafe_allow_html=True)

    if analyze and image is not None:
        with st.spinner("Analyzing fruit..."):
            try:
                result = predict_freshness(model, preprocess_image(image))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()
        st.session_state.last_result = (uploaded.name, result)
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "File": uploaded.name,
            "Prediction": result.label,
            "Confidence (%)": round(result.confidence, 2),
            "Fresh (%)": round(result.fresh_prob, 2),
            "Rotten (%)": round(result.rotten_prob, 2),
        })

    with right:
        st.subheader("2. Result")
        if st.session_state.last_result is None:
            st.markdown('<div class="card"><p class="step">Your result will appear here after '
                        "you upload an image and press <b>Analyze Fruit</b>.</p></div>",
                        unsafe_allow_html=True)
        else:
            name, result = st.session_state.last_result
            st.caption(f"File: {name}")
            show_result(result)
            report = pd.DataFrame([{
                "File": name, "Prediction": result.label,
                "Confidence (%)": round(result.confidence, 2),
                "Fresh (%)": round(result.fresh_prob, 2),
                "Rotten (%)": round(result.rotten_prob, 2),
                "Raw sigmoid score": round(result.raw_score, 6),
            }])
            st.download_button("Download this report (CSV)", report.to_csv(index=False),
                               file_name="prediction_report.csv", mime="text/csv")


def page_model_info():
    st.title("Model Information")

    cols = st.columns(4)
    for col, (label, value) in zip(cols, METRICS.items()):
        col.markdown(metric_card(label, value), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("CNN architecture")
        st.markdown(
            "| Layer | Details |\n|---|---|\n"
            "| Input | 224 × 224 × 3 (RGB) |\n"
            "| Conv2D + MaxPool | 32 filters, 3×3, ReLU, 2×2 pool |\n"
            "| Conv2D + MaxPool | 64 filters, 3×3, ReLU, 2×2 pool |\n"
            "| Conv2D + MaxPool | 128 filters, 3×3, ReLU, 2×2 pool |\n"
            "| Flatten | - |\n"
            "| Dense | 128 units, ReLU |\n"
            "| Output | 1 unit, Sigmoid |"
        )
        st.caption("Optimizer: Adam · Loss: binary cross-entropy · Augmentation: flip, "
                   "rotation, zoom · Early stopping on validation loss.")
    with c2:
        st.subheader("Data and task")
        st.markdown(
            "- **Input size:** 224 × 224 pixels, pixel values scaled to 0-1 (÷ 255)\n"
            "- **Original dataset classes:** 6 (fresh/rotten × apples, bananas, oranges)\n"
            "- **Task:** binary classification\n"
            "  - Label **0 = Fresh** (freshapples, freshbanana, freshoranges)\n"
            "  - Label **1 = Rotten** (rottenapples, rottenbanana, rottenoranges)\n"
            "- **Output:** sigmoid probability of *Rotten*; threshold 0.5\n"
            "- The model does **not** identify the fruit type."
        )

    st.subheader("Limitations")
    st.warning(
        "The reported accuracy is measured on the dataset's test split. It is not a "
        "guarantee for real-world photos. The model was trained on only three fruit "
        "types and may behave unpredictably on other fruits, cluttered backgrounds, "
        "or unusual lighting."
    )


def page_history():
    st.title("Prediction History")
    st.caption("Predictions made in this browser session.")
    history = st.session_state.history
    if not history:
        st.info("No predictions yet. Go to **Fruit Detection** to analyze an image.")
        return

    df = pd.DataFrame(history)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total predictions", len(df))
    c2.metric("Fresh", int((df["Prediction"] == "Fresh").sum()))
    c3.metric("Rotten", int((df["Prediction"] == "Rotten").sum()))

    st.dataframe(df, use_container_width=True, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("Download history (CSV)", df.to_csv(index=False),
                       file_name="prediction_history.csv", mime="text/csv",
                       use_container_width=True)
    if d2.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    init_state()
    st.sidebar.title("🍎 Fruit Freshness AI")
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Fruit Detection", "Model Information", "Prediction History"],
    )
    st.sidebar.divider()
    st.sidebar.caption("Custom CNN · TensorFlow/Keras · Streamlit")

    {
        "Home": page_home,
        "Fruit Detection": page_detection,
        "Model Information": page_model_info,
        "Prediction History": page_history,
    }[page]()


main()
