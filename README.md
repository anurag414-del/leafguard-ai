# 🌿 LeafGuard AI

**AI-powered plant leaf disease detection.** Upload a photo of a leaf and LeafGuard AI tells you the crop, the most likely disease, how confident the model is, and what to look for and do about it.

LeafGuard AI recognises **21 conditions across 4 crops** (Apple, Corn, Potato and Tomato) using a MobileNetV2 deep-learning model served through a Flask web app.

> **Disclaimer:** LeafGuard AI gives an AI estimate, not a professional diagnosis. For treatment decisions, confirm with a local agricultural extension officer or plant pathologist.

---

## Table of contents

- [Features](#features)
- [Supported crops and conditions](#supported-crops-and-conditions)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Model details](#model-details)
- [Results](#results)
- [API](#api)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Future improvements](#future-improvements)

---

## Features

- Detects **21 plant conditions** (diseases, a pest problem, and healthy leaves) for 4 crops
- Drag-and-drop image upload with live preview
- Confidence score with an animated ring, plus the **top 3 predictions**
- Built-in knowledge base: a short explanation, symptoms to look for, and recommended actions for every condition
- "Uncertain" warning when the model's confidence is below 60%
- Input validation: PNG/JPG only, maximum 5 MB
- Responsive, dark-themed interface

## Supported crops and conditions

| Crop | Conditions |
|---|---|
| 🍎 Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| 🌽 Corn | Gray leaf spot (Cercospora), Common rust, Northern leaf blight, Healthy |
| 🥔 Potato | Early blight, Late blight, Healthy |
| 🍅 Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Mosaic virus, Yellow leaf curl virus, Healthy |

## How it works

```
Leaf photo → resize to 128×128 → MobileNetV2 model → 21 class probabilities
          → top result + confidence + top 3 → web page with disease information
```

1. The user uploads a leaf image on the web page.
2. The Flask server (`app.py`) resizes it to 128×128 and passes raw 0–255 pixel values to the model (rescaling is built into the model).
3. The model outputs a probability for each of the 21 classes.
4. The page shows the best match, the confidence, the top 3 alternatives and information about the condition. Results below 60% confidence are marked as *Uncertain*.

## Project structure

```
LeafGaurdAI/
├── app.py                      # Flask web server and /predict API
├── train_model.py              # Trains and evaluates the model, writes labels.json
├── evaluate_model.py           # Classification report and confusion matrix
├── predict.py                  # Command-line prediction for a single image
├── check_dataset.py            # Counts images per class
├── labels.json                 # Class names in the order the model uses
├── leafguard_model.keras       # Final trained model
├── best_leafguard_multi.keras  # Best checkpoint saved during training
├── templates/
│   └── index.html              # Web page
├── static/
│   ├── style.css               # Styling
│   └── script.js               # Front-end logic and disease knowledge base
├── test_images/                # Sample images for testing
└── dataset/                    # Training data (not included, see below)
```

## Getting started

### Requirements

- Python 3.10 or newer
- The packages below

```
tensorflow
flask
pillow
numpy
scikit-learn
matplotlib
seaborn
```

### Installation

```bash
# 1. Clone or download the project, then enter the folder
cd LeafGaurdAI

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install the dependencies
pip install tensorflow flask pillow numpy scikit-learn matplotlib seaborn
```

> On Windows, use `py` instead of `python` if `python` is not recognised.

### Dataset

The model is trained on the **PlantVillage** dataset (21 of its classes: Apple, Corn, Potato and Tomato). The dataset is not included in this repository because of its size.

Place the class folders directly inside `dataset/`, one folder per class, named `Crop___Disease` (three underscores):

```
dataset/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
├── Corn_(maize)___Common_rust_/
├── Corn_(maize)___Northern_Leaf_Blight/
├── Corn_(maize)___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── Tomato___Bacterial_spot/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___Leaf_Mold/
├── Tomato___Septoria_leaf_spot/
├── Tomato___Spider_mites Two-spotted_spider_mite/
├── Tomato___Target_Spot/
├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
├── Tomato___Tomato_mosaic_virus/
└── Tomato___healthy/
```

Do not nest the folders (for example `dataset/PlantVillage/...`). Verify the layout with:

```bash
python check_dataset.py
```

The dataset used here has **21 classes and 21,866 images**. The classes are uneven (for example Tomato Yellow Leaf Curl Virus has 4,286 images, Potato healthy only 121), which the training script compensates for with class weights.

## Usage

### 1. Train the model

```bash
python train_model.py
```

This reads the class names from the `dataset/` folders, trains in two phases, and saves `leafguard_model.keras` and `labels.json`. On a CPU, expect roughly an hour or more; a free Google Colab GPU is much faster. Keep the laptop plugged in and awake while it runs.

### 2. Evaluate the model

```bash
python evaluate_model.py
```

Prints a per-class classification report and shows a confusion matrix.

### 3. Predict from the command line

```bash
python predict.py test_images/leaf1.jpg
```

Prints the top 5 predictions with their probabilities.

### 4. Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, upload a leaf image and press **Analyze leaf**.

> `app.py` needs `leafguard_model.keras` and `labels.json` in the same folder, so run `train_model.py` first if they are missing.

## Model details

| Item | Value |
|---|---|
| Architecture | MobileNetV2 (ImageNet pretrained) + global average pooling + dropout (0.3) + 21-way softmax |
| Input size | 128 × 128 RGB, raw 0–255 pixels (rescaling to [-1, 1] is inside the model) |
| Data split | 80% training / 20% validation (seed 123) |
| Augmentation | Random flip, rotation (0.1), zoom (0.1), brightness (0.1), contrast (0.1), active only during training |
| Class weights | Square root of the inverse class frequency, to help small classes without over-predicting them |
| Phase 1 | Base model frozen, train the new head. Adam, learning rate 1e-3, up to 10 epochs |
| Phase 2 | Last 80 base layers unfrozen. Adam, learning rate 3e-5, up to 25 epochs |
| Callbacks | Model checkpoint, early stopping (patience 5), learning-rate reduction on plateau |

## Results

Measured on the 4,373-image validation split:

- **Validation accuracy: 95.70%** (validation loss 0.1468)
- **Macro F1: 0.95**, weighted F1: 0.96

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Apple___Apple_scab | 0.89 | 1.00 | 0.94 | 96 |
| Apple___Black_rot | 1.00 | 1.00 | 1.00 | 80 |
| Apple___Cedar_apple_rust | 1.00 | 0.97 | 0.99 | 38 |
| Apple___healthy | 1.00 | 0.99 | 0.99 | 251 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.79 | 0.99 | 0.88 | 82 |
| Corn___Common_rust | 1.00 | 1.00 | 1.00 | 199 |
| Corn___Northern_Leaf_Blight | 0.96 | 0.88 | 0.92 | 170 |
| Corn___healthy | 1.00 | 1.00 | 1.00 | 176 |
| Potato___Early_blight | 0.97 | 0.99 | 0.98 | 156 |
| Potato___Late_blight | 0.94 | 0.97 | 0.95 | 163 |
| Potato___healthy | 0.90 | 1.00 | 0.95 | 27 |
| Tomato___Bacterial_spot | 0.94 | 0.98 | 0.96 | 321 |
| Tomato___Early_blight | 1.00 | 0.77 | 0.87 | 166 |
| Tomato___Late_blight | 0.92 | 0.96 | 0.94 | 306 |
| Tomato___Leaf_Mold | 0.98 | 0.92 | 0.95 | 138 |
| Tomato___Septoria_leaf_spot | 0.97 | 0.93 | 0.95 | 279 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.92 | 0.91 | 0.91 | 267 |
| Tomato___Target_Spot | 0.83 | 0.96 | 0.89 | 217 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 1.00 | 0.98 | 0.99 | 888 |
| Tomato___Tomato_mosaic_virus | 1.00 | 0.82 | 0.90 | 72 |
| Tomato___healthy | 0.98 | 0.99 | 0.99 | 281 |

The weakest classes are Tomato Early blight (recall 0.77), Corn Gray leaf spot (precision 0.79), Tomato Target spot (precision 0.83) and Tomato mosaic virus (recall 0.82).

> These figures are for PlantVillage-style images. See [Known limitations](#known-limitations) for how they translate to real-world photos.

## API

### `POST /predict`

Send an image as `multipart/form-data` with the field name `image` (PNG or JPG, max 5 MB).

**Example response**

```json
{
  "prediction": "Tomato - Late blight",
  "confidence": 97.31,
  "top3": [
    { "label": "Tomato - Late blight", "confidence": 97.31 },
    { "label": "Tomato - Early blight", "confidence": 1.52 },
    { "label": "Potato - Late blight", "confidence": 0.64 }
  ]
}
```

If confidence is below 60%, `prediction` is returned as `Uncertain (closest: <label>)`.

**Errors** are returned as JSON with an `error` message and an HTTP status of 400 (bad or missing image), 413 (file too large) or 500 (server error).

## Known limitations

- **Real-world photos:** the model was trained on clean, uniform, lab-style leaf images. Accuracy is high on similar images and can be noticeably lower on cluttered field or internet photos.
- **Optimistic validation:** PlantVillage contains many near-duplicate images of the same leaves, so a random split makes validation easier than real use.
- **No "not a leaf" class:** non-leaf images can still receive a confident but meaningless prediction.
- **Small classes:** Potato healthy has only 27 validation images, so its scores are noisy.
- **Image size:** at 128 × 128, small spots that separate similar tomato diseases can be hard to see.
- **General guidance only:** the advice shown for each condition is general and is not a substitute for expert diagnosis.

For the best results, upload a close-up of **a single leaf** on a **plain background**, in good light and in focus.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model has N outputs but labels.json has M classes` | The model and `labels.json` come from different training runs. Retrain, or use the matching pair of files. |
| `FileNotFoundError` for the model or labels | Run `python train_model.py` first so that `leafguard_model.keras` and `labels.json` exist. |
| The web page looks old or broken after editing | Stop the server (`Ctrl + C`), restart `python app.py`, then hard refresh the browser (`Ctrl + F5`). Make sure no other copy of the app is running on port 5000. |
| `python` is not recognised on Windows | Use `py` instead (for example `py app.py`). |
| Training is very slow | Use a GPU, for example free Google Colab, or reduce the number of epochs. |
| Predictions are wrong on internet photos | Expected with this training data. See [Known limitations](#known-limitations) and [Future improvements](#future-improvements). |

## Future improvements

- Add real-world photos (for example from PlantDoc or your own phone shots) and keep a separate real-photo test set
- Add an "Other / not a leaf" class so random uploads are rejected
- Train at 224 × 224 resolution on a GPU
- Support more crops, possibly with a two-stage design (species classifier, then a disease model per species)
- Show Grad-CAM heatmaps to highlight where the model is looking
- Deploy the app online

## Credits

- Dataset: PlantVillage (Hughes and Salathé), widely mirrored on Kaggle
- Model: MobileNetV2 (Sandler et al.), via TensorFlow / Keras
- Built with Python, TensorFlow, Flask, Pillow, NumPy, scikit-learn, matplotlib and seaborn

---

*Made by [Your Name] as a mini project. For educational use.*
