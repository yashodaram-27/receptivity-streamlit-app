import streamlit as st
from tensorflow.keras.modeles import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image

# Load your trained model
model = load_model("receptivity_efficientnet_model.h5")

# Set class labels
class_labels = ['Post_receptive', 'pre_Receptive', 'receptive']

# Streamlit title
st.title("Endometrial Receptivity Prediction")
st.write("Upload a grayscale ultrasound image of the uterus (sagittal view).")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write("### Prediction:")
    st.write(f"**{class_labels[predicted_class]}**")

