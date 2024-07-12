import os
import numpy as np
from flask import Flask, request, render_template, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

model = load_model('/workspaces/eye/model.h5')

def load_and_preprocess_image(img_path, target_size):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array, img

def overlay_text_on_image(img, text):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text_position = (10, 10)  # Position at the top-left corner
    draw.text(text_position, text, font=font, fill="red")
    return img

def model_predict(img_path, model):
    img_array, img = load_and_preprocess_image(img_path, (256, 256))  # Resize the image to 256x256
    prediction = model.predict(img_array)
    
    predicted_class_index = np.argmax(prediction)
    class_names = ['Healthy', 'Glaucoma', 'Cataract']  # Replace with your actual class names

    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
    else:
        predicted_class = 'Diabetic_Retinopathy'

    img_with_text = overlay_text_on_image(img, predicted_class)
    
    modified_img_path = os.path.join('uploads', 'predicted_' + os.path.basename(img_path))
    img_with_text.save(modified_img_path)

    return modified_img_path, predicted_class

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            modified_img_path, predicted_class = model_predict(file_path, model)
            return render_template('result.html', result=predicted_class, img_path=modified_img_path)
    return None

@app.route('/uploads/<filename>')
def send_file_from_uploads(filename):
    return send_file(os.path.join('uploads', filename))

if __name__ == '__main__':
    app.run(debug=True)
