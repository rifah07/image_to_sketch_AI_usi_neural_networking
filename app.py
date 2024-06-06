from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Ensure upload and result folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load pre-trained model from TensorFlow Hub
MODEL_URL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
LOCAL_MODEL_PATH = None  # Set this to the path if you have a local model

if LOCAL_MODEL_PATH:
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
else:
    model = hub.load(MODEL_URL)

def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    img = img.convert('RGB')
    img.thumbnail((max_dim, max_dim))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def convert_to_sketch(image_path):
    content_image = load_img(image_path)
    # Using the same content image as style image for sketch effect
    style_image = load_img(image_path)
    outputs = model(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    stylized_image = tf.image.convert_image_dtype(stylized_image, tf.uint8)
    stylized_image = np.array(stylized_image[0])
    
    # Convert the stylized image to a PIL image
    sketch = Image.fromarray(stylized_image)
    
    # Convert the image to black and white
    sketch_bw = ImageOps.grayscale(sketch)
    
    # Convert PIL image to numpy array for OpenCV processing
    sketch_bw_np = np.array(sketch_bw)
    
    # Apply Canny edge detection
    edges = cv2.Canny(sketch_bw_np, 100, 200)
    
    # Invert colors
    edges_inverted = cv2.bitwise_not(edges)
    
    # Convert edges back to a PIL image
    edges_pil = Image.fromarray(edges_inverted)
    
    return edges_pil

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        print('No image part')
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        print('No selected file')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Convert the image to sketch using neural network
        sketch = convert_to_sketch(filepath)
        sketch_path = os.path.join(app.config['RESULT_FOLDER'], f'sketch_{filename}')
        sketch.save(sketch_path)
        
        sketch_url = url_for('static', filename=f'results/sketch_{filename}')
        print(f'Sketch URL: {sketch_url}')  # Debug print
        return render_template('index.html', sketch_url=sketch_url)

if __name__ == '__main__':
    app.run(debug=True)
