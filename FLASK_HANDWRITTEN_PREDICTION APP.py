from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('handwritten_digit_model.h5')

def preprocess_image(image):
    # Your image preprocessing logic here
    # For example, resize the image to the required input shape
    img = image.resize((28, 28))
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.reshape((1, 28, 28, 1))  # Add batch dimension
    return img_array

def predict_digit(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    digit = np.argmax(prediction)
    return digit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        img = Image.open(io.BytesIO(file.read())).convert("L")  # Convert to grayscale
        digit = predict_digit(img)
        return render_template('result.html', digit=digit)
    except Exception as e:
        print(f"Error processing image: {e}")
        return render_template('index.html', error='Error processing image')

if __name__ == '__main__':
    app.run(debug=True)
