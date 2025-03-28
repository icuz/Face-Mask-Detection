import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide"
)

# Custom styles
st.markdown("""
    <style>
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stFileUploader > div {
        text-align: center;
    }
    .stImage {
        border-radius: 10px;
    }
    .reportview-container {
        background: linear-gradient(to right, #ffafbd, #ffc3a0);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("logo.png", use_container_width=True)  # Replace with your logo
# st.sidebar.title("Navigation")

# Load Models (Caching for faster load)
@st.cache_resource
def load_mask_model():
    return load_model('best_model.keras')

@st.cache_resource
def load_face_detector():
    model_path = "deploy.prototxt"
    weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
    return cv2.dnn.readNetFromCaffe(model_path, weights_path)

best_model = load_mask_model()
face_net = load_face_detector()

# Label mapping
label_map = {0: 'With Mask 😷', 1: 'Without Mask ❌', 2: 'Mask Worn Incorrectly ⚠️'}

def detect_and_predict(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Higher confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY, endX, endY = max(0, startX), max(0, startY), min(w, endX), min(h, endY)

            face = image[startY:endY, startX:endX]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)

            label = best_model.predict(face)
            label_idx = np.argmax(label, axis=1)[0]

            color = (0, 255, 0) if label_idx == 0 else (0, 0, 255)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 3)
            cv2.putText(image, label_map[label_idx], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Main UI
st.title("🛡️ ML-Powered Face Mask Detection")
st.markdown("**Upload an image to determine if a mask is being worn correctly.**")

uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Progress bar
    progress_bar = st.progress(0)
    for percent in range(100):
        progress_bar.progress(percent + 1)

    processed_image = detect_and_predict(image)
    
    st.success("✅ Detection Complete!")
    st.image(processed_image, caption="🔍 Detected Faces", use_container_width=True)
