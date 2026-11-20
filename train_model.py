import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# ==========================================
# 1. Configuration
# ==========================================

DATASET = "dataset"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 123
EPOCHS_FROZEN = 10
EPOCHS_FINETUNE = 25

MODEL_PATH = "leafguard_model.keras"
BEST_MODEL_PATH = "best_leafguard_multi.keras"
LABELS_PATH = "labels.json"

tf.keras.utils.set_random_seed(SEED)

# Class names are read from the dataset folders automatically
class_names = sorted(
    d for d in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, d))
)
num_classes = len(class_names)

print(f"Found {num_classes} classes:")
for c in class_names:
    print("  ", c)

# Save the class order so app.py / predict.py / evaluate_model.py use the same one
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# ==========================================
# 2. Datasets
# (both use the default shuffle=True with the same seed,
#  so validation is exactly the other 20% of the training split)
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET, class_names=class_names,
    validation_split=0.2, subset="training", seed=SEED,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET, class_names=class_names,
    validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int"
)

# ==========================================
# 3. Class weights (counted from file paths - fast)
# ==========================================

name_to_idx = {n: i for i, n in enumerate(class_names)}
counts = np.zeros(num_classes)
for p in train_ds.file_paths:
    counts[name_to_idx[os.path.basename(os.path.dirname(p))]] += 1

if np.any(counts == 0):
    raise ValueError("At least one class has no training images.")

total = counts.sum()
# Square root softens the weights so tiny classes don't become "dump classes"
class_weights = {
    i: float((total / (num_classes * counts[i])) ** 0.5)
    for i in range(num_classes)
}

print("\nTraining images per class:")
for i, c in enumerate(class_names):
    print(f"  {c:55s} {int(counts[i]):6d}  weight={class_weights[i]:.2f}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# 4. Model (MobileNetV2 transfer learning)
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
x = data_augmentation(inputs)                                  # raw 0-255 pixels
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)     # -> [-1, 1]
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# ==========================================
# 5. Callbacks
# ==========================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
)  # same object for both phases, so it keeps the true best


def make_callbacks():
    return [
        checkpoint,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-7, verbose=1),
    ]


# ==========================================
# 6. Phase 1: train the new head
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== Phase 1: training new top layers ===")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN,
          class_weight=class_weights, callbacks=make_callbacks())

# ==========================================
# 7. Phase 2: fine-tune top of MobileNetV2
# ==========================================

base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== Phase 2: fine-tuning ===")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
          class_weight=class_weights, callbacks=make_callbacks())

# ==========================================
# 8. Save + evaluate
# ==========================================

best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
best_model.save(MODEL_PATH)

val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
print(f"\nValidation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_acc * 100:.2f}%")

y_true, y_pred = [], []
for images, labels in val_ds:
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(best_model.predict(images, verbose=0), axis=1).tolist())

print("\nClassification Report:\n")
print(classification_report(
    y_true, y_pred, labels=list(range(num_classes)),
    target_names=class_names, zero_division=0))

print("Saved:", MODEL_PATH, "and", LABELS_PATH)