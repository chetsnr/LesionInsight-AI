from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load your trained CNN model
model = load_model("/Users/anugrah/Downloads/my_model.h5")  # Ensure my_model.h5 is in the project root directory

# Define class labels based on your model's output
CLASS_LABELS = [
    "Acne and Rosacea Photos",
    "Atopic Dermatitis Photos",
    "Eczema Photos",
    "Hair Loss Photos (Alopecia and other Hair Diseases)",
    "Nail Fungus and other Nail Disease",
    "Psoriasis pictures (Lichen Planus and related diseases)"
]

# Upload folder configuration
UPLOAD_FOLDER = 'static/images/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Routes
@app.route('/')
def home():
    """Home page with a video background"""
    return render_template('index.html')

@app.route('/upload')
def upload():
    """Page for uploading an image"""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and predict the disease"""
    # Check if a file is in the request
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image for model prediction
    img = tf.keras.utils.load_img(filepath, target_size=(224, 224))  # Resize to model input size
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class_index = predictions.argmax()  # Get the index of the highest probability
    predicted_class = CLASS_LABELS[predicted_class_index]  # Map index to class label

    # Pass the result and image to the result template
    return render_template('result.html', result=predicted_class, filename=filename)

@app.route('/team')
def team():
    """Page for displaying the team information"""
    return render_template('team.html')  # Ensure you have a team.html template

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
