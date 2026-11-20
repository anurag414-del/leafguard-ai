
import os
import re
import json

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError


# =====================================
# 1. Flask Configuration
# =====================================

app = Flask(__name__)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 60.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "leafguard_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")


# =====================================
# 2. Load Model and Class Names
# =====================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs but "
        f"labels.json has {len(class_names)} classes."
    )

print(f"\nLeafGuard AI loaded with {len(class_names)} classes.")


# =====================================
# 3. Format Class Names
# =====================================

def pretty(name):
    """
    Example:
    Corn_(maize)___Common_rust
    Returns:
    ('Corn', 'Common rust')
    """

    match = re.match(r"^(.*?)_{3,}(.*)$", name)

    if match:
        crop, disease = match.group(1), match.group(2)
    else:
        crop, disease = name, ""

    def clean(text):
        return " ".join(text.replace("_", " ").split())

    crop = clean(crop)
    disease = clean(disease)

    # Remove crop names inside brackets
    crop = re.sub(r"\s*\(.*?\)", "", crop).strip()

    if disease.lower() == "healthy":
        disease = "Healthy"
    elif disease:
        disease = disease[0].upper() + disease[1:]

    return crop, disease


display = [pretty(name) for name in class_names]


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def label_text(index):
    crop, disease = display[index]

    if crop.lower() == "other":
        return "Not a plant leaf"

    if not disease:
        return crop

    return f"{crop} - {disease}"


# =====================================
# 4. Disease Symptoms and Solutions
# =====================================

def get_disease_solution(crop, disease, is_confident=True):

    if not is_confident:
        return {
            "symptoms": (
                "The model is not confident enough "
                "to identify the disease."
            ),
            "solution": [
                "Upload a clear image of the affected leaf.",
                "Capture the leaf in good lighting.",
                "Consult an agricultural expert for confirmation."
            ],
            "prevention": "Continue monitoring the plant."
        }

    if crop.lower() == "other":
        return {
            "symptoms": (
                "The uploaded image may not contain "
                "a plant leaf."
            ),
            "solution": [
                "Upload a clear image of a plant leaf."
            ],
            "prevention": "Not applicable."
        }

    disease_key = disease.lower().strip()

    disease_data = {

        "healthy": {
            "symptoms": "No visible disease detected.",
            "solution": [
                "No disease-specific treatment is required.",
                "Continue proper watering and plant nutrition."
            ],
            "prevention": (
                "Regularly inspect leaves for early "
                "signs of disease."
            )
        },

        "early blight": {
            "symptoms": (
                "Brown circular spots with concentric rings, "
                "usually on older leaves."
            ),
            "solution": [
                "Remove and safely dispose of infected leaves.",
                "Avoid watering directly on leaves.",
                "Maintain proper spacing between plants.",
                "Consult an agricultural expert about "
                "approved fungicide treatment."
            ],
            "prevention": (
                "Practice crop rotation and use "
                "disease-free seeds."
            )
        },

        "late blight": {
            "symptoms": (
                "Dark, water-soaked patches that may spread "
                "rapidly across leaves."
            ),
            "solution": [
                "Remove and safely dispose of infected plant parts.",
                "Avoid overhead irrigation.",
                "Improve air circulation around plants.",
                "Consult an agricultural expert about "
                "suitable treatment."
            ],
            "prevention": (
                "Use healthy planting material and avoid "
                "excessive moisture."
            )
        },

        "common rust": {
            "symptoms": (
                "Small reddish-brown raised spots on leaves."
            ),
            "solution": [
                "Monitor the spread of infection.",
                "Remove infected plant debris after harvest.",
                "Use resistant crop varieties where available.",
                "Consult an agricultural expert if "
                "infection is severe."
            ],
            "prevention": (
                "Use resistant varieties and maintain "
                "field hygiene."
            )
        },

        "apple scab": {
            "symptoms": (
                "Dark olive or black spots on leaves and fruit."
            ),
            "solution": [
                "Remove fallen infected leaves.",
                "Prune branches to improve airflow.",
                "Avoid overhead watering.",
                "Follow locally approved disease-management advice."
            ],
            "prevention": (
                "Maintain orchard hygiene and use resistant "
                "varieties where available."
            )
        }
    }

    return disease_data.get(disease_key, {
        "symptoms": (
            f"Possible symptoms associated with {disease}."
        ),
        "solution": [
            "Inspect the affected leaves carefully.",
            "Remove severely infected plant parts if appropriate.",
            "Avoid excessive watering.",
            "Consult a local agricultural expert for "
            "disease-specific treatment."
        ],
        "prevention": (
            "Maintain plant hygiene and regularly "
            "monitor crop health."
        )
    })


# =====================================
# 5. Home Route
# =====================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# 6. Prediction Route
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether an image was uploaded
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    # Check empty filename
    if file.filename == "":
        return jsonify({
            "error": "Please select an image."
        }), 400

    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            "error": "Only PNG or JPG images are allowed."
        }), 400

    try:

        # Open and preprocess image
        image = (
            Image.open(file)
            .convert("RGB")
            .resize((128, 128))
        )

        arr = np.expand_dims(
            np.array(image, dtype=np.float32),
            axis=0
        )

        # Model prediction
        probs = model.predict(arr, verbose=0)[0]

        # Get top 3 predictions
        top = np.argsort(probs)[::-1][:3]

        best = int(top[0])

        confidence = float(probs[best] * 100)

        result = label_text(best)

        # Confidence threshold
        is_confident = confidence >= CONFIDENCE_THRESHOLD

        if not is_confident:
            result = f"Uncertain (closest: {result})"

        # Get crop and disease details
        crop, disease = display[best]

        # Get symptoms, solutions and prevention
        disease_info = get_disease_solution(
            crop,
            disease,
            is_confident
        )

        # Prepare response
        return jsonify({

            "prediction": result,

            "confidence": round(confidence, 2),

            "top3": [
                {
                    "label": label_text(int(i)),
                    "confidence": round(
                        float(probs[i] * 100), 2
                    )
                }
                for i in top
            ],

            "symptoms": disease_info["symptoms"],

            "solution": disease_info["solution"],

            "prevention": disease_info["prevention"]

        })

    except UnidentifiedImageError:

        return jsonify({
            "error": "That file is not a valid image."
        }), 400

    except Exception as e:

        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Please try again."
        }), 500


# =====================================
# 7. File Size Error Handler
# =====================================

@app.errorhandler(413)
def too_large(error):

    return jsonify({
        "error": "Image is too large (maximum 5 MB)."
    }), 413


# =====================================
# 8. Run Flask Application
# =====================================

if __name__ == "__main__":

    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1"
    )
