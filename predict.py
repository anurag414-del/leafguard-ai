import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("leafguard_model.keras")
with open("labels.json", encoding="utf-8") as f:
    class_names = json.load(f)

image_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/leaf1.jpg"

img = Image.open(image_path).convert("RGB").resize((128, 128))
arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)   # raw 0-255

probs = model.predict(arr, verbose=0)[0]
print("\nTop predictions:")
for i in np.argsort(probs)[::-1][:5]:
    print(f"  {class_names[i]:55s} {probs[i] * 100:6.2f}%")