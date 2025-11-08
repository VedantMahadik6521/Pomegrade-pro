import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("pomegranate_model.h5")

# Define labels
class_names = ['Healthy', 'Unhealthy']

# Streamlit UI
st.title("🍇 Pomegranate Leaf Health Classifier")
st.write("Upload an image of a pomegranate leaf to detect if it’s **Healthy** or **Unhealthy**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # same size used while training
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.markdown(f"### 🌿 Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
