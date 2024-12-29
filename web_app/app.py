# ./web_app/app.py
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect
from PIL import Image


app = Flask(__name__)

try:
    MODEL_PATH = './models/saved_models/mnist_model.keras'
    model = tf.keras.models.load_model(MODEL_PATH)
except ValueError as e:
    print(f"Error {e}.\nModel needs to be trained.")

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    threshold = 128
    img = img.point(lambda p: p > threshold and 255)
    img = img.resize((28, 28))

    img_array = np.array(img)
    img_array = center_image(img_array)
    img_array = normalise_image(img_array)

    return img_array

def normalise_image(img_array):
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def center_image(img_array):
    coords = np.column_stack(np.where(img_array > 0.1))
    if coords.size == 0:
        return img_array

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img_array[y_min:y_max+1, x_min:x_max+1]

    new_img = np.zeros((28, 28), dtype=img_array.dtype)

    h, w = cropped.shape
    y_offset = (28 - h) // 2
    x_offset = (28 - w) // 2
    new_img[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    return new_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join('./web_app/static', file.filename)
            file.save(filepath)

            base, ext = os.path.splitext(file.filename)
            png_filename = f"{base}.png"
            png_filepath = os.path.join('./web_app/static', png_filename)


            img = Image.open(filepath)
            img.save(png_filepath, 'PNG')

            img_array = preprocess_image(png_filepath)
            prediction = model.predict(img_array)
            predicted_label = np.argmax(prediction[0])

            return render_template(
                'results.html',
                filename=png_filename,
                label=predicted_label
            )

    return render_template('index.html')

@app.route('/show/<filename>')
def show_image(filename):
    return f"<img src='/static/{filename}' alt='User Uploaded Image'>"