# Brain-Tumor-Detection
The Brain Tumor Detection Application is a cutting-edge solution built on machine learning principles, specifically leveraging Convolutional Neural Networks (CNNs), to accurately distinguish between four types of brain tumors: Normal, Glioma, Meningioma, and Pituitary. 
Brain Tumor Detection with CNN

This repository implements a brain tumor detection system using Convolutional Neural Networks (CNNs) in Python and a user-friendly front-end. It can classify brain MRI scans into four categories: normal, pituitary tumor, meningioma, and glioma.

Project Structure:

backend/: Python code for CNN model training, prediction, and data processing.
model.py: Defines the CNN architecture and training process.
predict.py: Loads the trained model and performs predictions on MRI scans.
utils.py: Helper functions for data preprocessing and visualization.
dataset/: Training and testing datasets for the CNN model (replace with your data source).
frontend/: HTML, CSS, and JavaScript code for the user interface.
index.html: Main HTML template for the web application.
style.css: Stylesheet for the user interface.
script.js: JavaScript code for handling user interaction and communication with the backend.
Requirements:

Python 3.x (with libraries like TensorFlow, Keras, NumPy)
Node.js (for running a local server for the front-end)
