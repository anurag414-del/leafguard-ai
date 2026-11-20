"""
Run from the project folder:  python diagnose_model.py
Tells us WHY the model always predicts Early Blight.
"""
import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "leafguard_potato_model.keras"
DATASET = "dataset"
SAMPLES_PER_CLASS = 15

folders = ["Potato___Early_blight", "Potato___healthy", "Potato___Late_blight"]
names = ["Early Blight", "Healthy", "Late Blight"]

model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------------------------------------
# 1. Are the weights sane?
# ------------------------------------------------------------
print("\n=== 1. WEIGHT CHECK ===")
bad = False
for w, v in zip(model.weights, model.get_weights()):
    mx = float(np.abs(v).max())
    nan = bool(np.isnan(v).any())
    flag = ""
    if nan or mx > 50:
        flag = "  <-- SUSPICIOUS"
        bad = True
    print(f"{w.path:45s} absmax={mx:10.3f} nan={nan}{flag}")
print("Weights look exploded/NaN!" if bad else "Weights look normal.")


def load(path):
    im = Image.open(path).convert("RGB").resize((128, 128))
    return np.array(im, dtype=np.float32)


# ------------------------------------------------------------
# 2. Predict real dataset images (the app's way: raw 0-255)
# ------------------------------------------------------------
print("\n=== 2. REAL DATASET IMAGES (raw 0-255, same as app.py) ===")
random.seed(1)
for folder, true_name in zip(folders, names):
    files = [f for f in os.listdir(os.path.join(DATASET, folder))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
    batch = np.stack([load(os.path.join(DATASET, folder, f)) for f in files])
    probs = model.predict(batch, verbose=0)
    preds = probs.argmax(axis=1)
    counts = {n: int((preds == i).sum()) for i, n in enumerate(names)}
    correct = int((preds == names.index(true_name)).sum())
    print(f"Actual {true_name:13s}: {correct}/{len(files)} correct | predicted -> {counts}"
          f" | mean top conf {probs.max(axis=1).mean()*100:.1f}%")

# ------------------------------------------------------------
# 3. Same images but pre-scaled to 0-1 (to test scaling mismatch)
# ------------------------------------------------------------
print("\n=== 3. SAME TEST WITH /255 PRE-SCALING (scaling mismatch test) ===")
for folder, true_name in zip(folders, names):
    files = [f for f in os.listdir(os.path.join(DATASET, folder))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
    batch = np.stack([load(os.path.join(DATASET, folder, f)) for f in files]) / 255.0
    preds = model.predict(batch, verbose=0).argmax(axis=1)
    correct = int((preds == names.index(true_name)).sum())
    print(f"Actual {true_name:13s}: {correct}/{len(files)} correct")

# ------------------------------------------------------------
# 4. Junk inputs
# ------------------------------------------------------------
print("\n=== 4. JUNK INPUTS ===")
junk = {
    "black": np.zeros((128, 128, 3), np.float32),
    "white": np.full((128, 128, 3), 255, np.float32),
    "noise": np.random.randint(0, 256, (128, 128, 3)).astype(np.float32),
}
for k, v in junk.items():
    p = model.predict(v[None], verbose=0)[0]
    print(k, dict(zip(names, np.round(p * 100, 1))))

print("\nSend me this whole output.")