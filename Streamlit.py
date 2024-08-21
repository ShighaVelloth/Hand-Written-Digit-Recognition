import streamlit as st
import numpy as np
from keras.models import load_model
import cv2

# Load the trained CNN model
CNN = load_model("C:/project/handwritten_digit_model.h5") 

st.title('HandWritten Prediction App')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image as OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Preprocess the image
    resized_image = cv2.resize(opencv_image, (28, 28))  # Resize to 28x28 pixels
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    normalized_image = gray_image.astype('float32') / 255.0  # Normalize pixel values

    # Make prediction
    prediction = CNN.predict(np.expand_dims(normalized_image, axis=(0, -1)))
    predicted_label = np.argmax(prediction)

    st.image(resized_image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Digit: {predicted_label}")
