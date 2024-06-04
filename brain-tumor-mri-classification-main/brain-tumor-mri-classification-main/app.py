from flask import Flask, render_template, request, url_for, send_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('best_weights.keras')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150), color_mode="grayscale")
    img_array = img_to_array(img)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor /= 255.0
    return img_tensor

def predict_image_class(image_path):
    img_tensor = preprocess_image(image_path)
    predictions = model.predict(img_tensor)
    predicted_class_index = np.argmax(predictions[0])
    class_labels = {0: 'Normal', 1: 'Glioma', 2: 'Meningioma', 3: 'Pituitary'}
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, predictions[0]

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predicted_class, class_probabilities = predict_image_class(file_path)
            image_url = url_for('uploaded_file', filename=file.filename)
            return render_template('result.html', image_url=image_url, predicted_class=predicted_class, class_probabilities=class_probabilities)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
