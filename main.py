
from pathlib import Path
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))



# Change this path to the location of your dataset.
DATASET_DIR = Path("coin_dataset")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 30

EXPECTED_CLASSES = ["10_cent", "20_cent", "50_cent", "1_dollar", "unknown"]

for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Missing folder: {directory.resolve()}\n"
            "Create the dataset structure shown above or update DATASET_DIR."
        )

print("Dataset folder:", DATASET_DIR.resolve())



#------TRAINING-----


train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)

print("Detected classes:", CLASS_NAMES)

if CLASS_NAMES != sorted(EXPECTED_CLASSES):
    print(
        "Warning: Keras sorts folder names alphabetically. "
        "The detected classes are valid as long as all five expected folders are present."
    )

missing = sorted(set(EXPECTED_CLASSES) - set(CLASS_NAMES))
if missing:
    raise ValueError(f"Missing expected class folders: {missing}")




#------Inspect class balance------



def count_images_by_class(directory: Path):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".heic"}
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = directory / class_name
        counts[class_name] = sum(
            1 for file in class_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in valid_extensions
        )
    return counts

train_counts = count_images_by_class(TRAIN_DIR)
val_counts = count_images_by_class(VAL_DIR)
test_counts = count_images_by_class(TEST_DIR)

print("Training counts:", train_counts)
print("Validation counts:", val_counts)
print("Testing counts:", test_counts)

x = np.arange(len(CLASS_NAMES))
width = 0.25

plt.figure(figsize=(11, 5))
plt.bar(x - width, [train_counts[c] for c in CLASS_NAMES], width, label="Train")
plt.bar(x, [val_counts[c] for c in CLASS_NAMES], width, label="Validation")
plt.bar(x + width, [test_counts[c] for c in CLASS_NAMES], width, label="Test")
plt.xticks(x, CLASS_NAMES, rotation=25)
plt.ylabel("Number of images")
plt.title("Dataset class distribution")
plt.legend()
plt.tight_layout()
plt.show()



#-----DISPLAY SAMPLE IMAGES------


plt.figure(figsize=(12, 8))

for images, labels in train_ds.take(1):
    number_to_show = min(15, len(images))
    for i in range(number_to_show):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = int(np.argmax(labels[i].numpy()))
        plt.title(CLASS_NAMES[label_index])
        plt.axis("off")

plt.tight_layout()
plt.show()





