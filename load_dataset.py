import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# 1. Dataset path
# =========================

dataset_path = "dataset"

# Same order as train_model.py
folder_names = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight"
]

# =========================
# 2. Load training dataset
# =========================

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    class_names=folder_names,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# =========================
# 3. Load validation dataset
# =========================

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    class_names=folder_names,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# =========================
# 4. Print class names
# =========================

class_names = train_ds.class_names

print("\nClass names:")
print(class_names)

# =========================
# 5. Print dataset sizes
# =========================

print("\nTraining batches:", len(train_ds))
print("Validation batches:", len(val_ds))

# =========================
# 6. Display images
# =========================

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")          # <-- fixed syntax error

plt.tight_layout()
plt.show()