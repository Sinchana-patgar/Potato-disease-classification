import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = 256
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "unknown"]

CLASS_INFO = {
    "Potato___Early_blight": {
        "label": "Early Blight",
        "color": "#f59e0b",
        "emoji": "🟡",
        "desc": "Caused by Alternaria solani. Look for dark brown spots with concentric rings on lower leaves.",
        "action": "Apply copper-based fungicide. Remove affected leaves early.",
    },
    "Potato___Late_blight": {
        "label": "Late Blight",
        "color": "#ef4444",
        "emoji": "🔴",
        "desc": "Caused by Phytophthora infestans. Water-soaked lesions that turn brown/black rapidly.",
        "action": "Apply mancozeb or chlorothalonil. Ensure good drainage.",
    },
    "Potato___healthy": {
        "label": "Healthy",
        "color": "#22c55e",
        "emoji": "🟢",
        "desc": "No signs of disease detected. The plant appears healthy.",
        "action": "Continue regular monitoring and preventive care.",
    },
    "unknown": {
        "label": "Unknown",
        "color": "#6b7280",
        "emoji": "⚪",
        "desc": "This image does not appear to be a potato leaf.",
        "action": "Please upload a clear image of a potato leaf.",
    },
}

# ── Model loading ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    keras_files = [
        f for f in os.listdir(model_dir)
        if f.endswith(".keras") and f.split(".")[0].isdigit()
    ]
    if not keras_files:
        st.error("No model found in the 'models/' directory.")
        st.stop()
    latest = max(keras_files, key=lambda f: int(f.split(".")[0]))
    model_path = os.path.join(model_dir, latest)
    return tf.keras.models.load_model(model_path), latest


def predict(model, image: Image.Image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    pred_class = CLASS_NAMES[idx]

    if pred_class == "unknown" or confidence < 70:
        return "unknown", confidence, predictions[0]

    return pred_class, confidence, predictions[0]


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Potato Blight Disease Detector",
    page_icon="🥔",
    layout="centered",
)

st.title("Potato Blight Disease Detector")
st.caption("Upload a leaf image to detect Early Blight, Late Blight, or Healthy status.")

with st.spinner("Loading model…"):
    model, model_file = load_model()
st.success(f"Model loaded: `{model_file}`", icon="✅")

st.divider()

uploaded = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG and PNG formats.",
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            pred_class, confidence, all_probs = predict(model, image)

        if pred_class == "unknown":
            st.warning("⚠️ This doesn't look like a potato leaf! Please upload a clear potato leaf image.")
        else:
            info = CLASS_INFO[pred_class]
            st.markdown(f"### {info['emoji']} {info['label']}")
            st.markdown(
                f"<span style='font-size:2rem; font-weight:600; color:{info['color']}'>"
                f"{confidence:.1f}% confidence</span>",
                unsafe_allow_html=True,
            )
            st.markdown("**About**")
            st.info(info["desc"])
            st.markdown("**Recommended action**")
            st.success(info["action"])

    st.divider()

    st.subheader("Class probabilities")
    display_classes = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    prob_data = {
        CLASS_INFO[c]["label"]: float(p) * 100
        for c, p in zip(CLASS_NAMES, all_probs)
        if c in display_classes
    }
    st.bar_chart(prob_data)

else:
    st.info("Please upload an image to get started.", icon="📂")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: CNN trained on PlantVillage dataset · 4 classes · 2027+ images")
