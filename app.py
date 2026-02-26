from flask import Flask, render_template, request
import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


# CONFIGURATION

CONFIDENCE_THRESHOLD = 70  

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# LOAD TRAINED MODEL
model = load_model("model/crop_disease_model.h5")
print("✅ Model output shape:", model.output_shape)

# LOAD CLASS LABELS


with open("model/class_labels.json", "r") as f:
    class_indices = json.load(f)


# Reverse mapping
class_labels = {int(v): k for k, v in class_indices.items()}
NUM_CLASSES = len(class_labels)

print("✅ Loaded classes:", class_labels)


# IMAGE PREPROCESSING FUNCTION

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ROUTES

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "❌ No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "❌ No file selected"

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess image
    img = preprocess_image(file_path)

    # Model prediction
    prediction = model.predict(img)
    predicted_index = int(np.argmax(prediction))
    confidence = round(float(np.max(prediction)) * 100, 2)

    predicted_class = class_labels[predicted_index]


    # CONFIDENCE-BASED DECISION

 
    if confidence < CONFIDENCE_THRESHOLD:
        message = "Healthy"
    else:
        message = f"🌿 Disease Detected: {predicted_class}"

    return render_template(
        "result.html",
        message=message,
        confidence=confidence,
        image_path=file_path
    )


# RUN FLASK APP

if __name__ == "__main__":
    app.run(debug=True)