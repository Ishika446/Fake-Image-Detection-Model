from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load model
print("Loading model...")
model = load_model(r"C:\Users\Ishika\Fake image detector\fake_vs_real_model(2).keras")
print("Model loaded successfully!")

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save and preprocess image
            filepath = os.path.join('static/uploads', secure_filename(file.filename))
            file.save(filepath)

            img = Image.open(filepath).convert('RGB').resize((128, 128))
            img_array = np.array(img)[:, :, :3]  # Ensure it's 3 channels (RGB)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = 'Fake' if prediction[0][0] > 0.5 else 'Real'

            return render_template('index1.html', result=result)

    return render_template('index1.html')

if __name__ == "__main__":
    app.run(debug=True)
