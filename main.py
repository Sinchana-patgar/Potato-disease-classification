import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ── Config ─────────────────────────────────────
IMAGE_SIZE = 256

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

CLASS_INFO = {
    "Potato___Early_blight": {
        "label": "Early Blight",
        "color": "#f59e0b",
        "emoji": "🟡",
        "desc": "Caused by Alternaria solani. Dark brown spots with concentric rings.",
        "action": "Apply copper-based fungicide. Remove infected leaves.",
    },
    "Potato___Late_blight": {
        "label": "Late Blight",
        "color": "#ef4444",
        "emoji": "🔴",
        "desc": "Caused by Phytophthora infestans. Rapid black/brown lesions.",
        "action": "Use mancozeb or chlorothalonil. Improve drainage.",
    },
    "Potato___healthy": {
        "label": "Healthy",
        "color": "#22c55e",
        "emoji": "🟢",
        "desc": "No disease detected.",
        "action": "Maintain regular care.",
    },
}

# ── Load Model ─────────────────────────────────
@st.cache_resource
def load_model():
    try:
        st.write("Loading model...")  # DEBUG

        model_dir = "models"

        if not os.path.exists(model_dir):
            st.error(f"{model_dir} folder not found.")
            st.stop()

        keras_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]

        st.write("Found files:", keras_files)  # DEBUG

        if len(keras_files) == 0:
            st.error("No .keras model found inside models/ folder.")
            st.stop()

        keras_files.sort(
            key=lambda x: os.path.getmtime(os.path.join(model_dir, x)),
            reverse=True
        )

        model_file = keras_files[0]
        model_path = os.path.join(model_dir, model_file)

        st.write("Loading:", model_path)  # DEBUG

        model = tf.keras.models.load_model(model_path)

        st.write("Model loaded successfully!")  # DEBUG

        return model, model_file

    except Exception as e:
        st.error("Model loading failed!")
        st.exception(e)
        st.stop()
# ── Prediction ─────────────────────────────────
def predict(model, image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    pred_class = CLASS_NAMES[idx]

    if confidence < 70:
        return None, confidence, predictions[0]

    return pred_class, confidence, predictions[0]

# ── UI ─────────────────────────────────────────
st.set_page_config(
    page_title="Potato Blight Disease Detector",
    page_icon="🥔",
    layout="centered"
)

st.title("🥔 Potato Blight Disease Detector")
st.caption("Upload a potato leaf image to detect disease.")

with st.spinner("Loading model..."):
    model, model_file = load_model()

st.success(f"Model loaded: {model_file}")
st.divider()

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            pred_class, confidence, probs = predict(model, image)

        if pred_class is None:
            st.warning("This doesn't look like a potato leaf! Please upload a clear potato leaf image.")
        else:
            info = CLASS_INFO[pred_class]
            st.markdown(f"### {info['emoji']} {info['label']}")
            st.markdown(
                f"<h2 style='color:{info['color']}'>{confidence:.2f}% confidence</h2>",
                unsafe_allow_html=True
            )
            st.markdown("**About**")
            st.info(info["desc"])
            st.markdown("**Recommended Action**")
            st.success(info["action"])

    st.divider()

    st.subheader("Class Probabilities")
    prob_data = {
        CLASS_INFO[c]["label"]: float(p) * 100
        for c, p in zip(CLASS_NAMES, probs)
    }
    st.bar_chart(prob_data)

else:
    st.info("Upload an image to begin.")

st.divider()
st.caption("CNN Model · 3 classes · PlantVillage dataset")
