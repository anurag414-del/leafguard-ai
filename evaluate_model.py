import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

model = tf.keras.models.load_model("leafguard_model.keras")
with open("labels.json", encoding="utf-8") as f:
    class_names = json.load(f)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset", class_names=class_names,
    validation_split=0.2, subset="validation", seed=123,
    image_size=(128, 128), batch_size=32, label_mode="int"
)

y_true, y_pred = [], []
for images, labels in val_ds:
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(model.predict(images, verbose=0), axis=1).tolist())

labels_idx = list(range(len(class_names)))
print(classification_report(
    y_true, y_pred, labels=labels_idx,
    target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.title("LeafGuard AI - Confusion Matrix")
plt.tight_layout()
plt.show()