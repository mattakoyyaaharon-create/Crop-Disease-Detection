import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# CONFIGURATION
# -------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 16
TEST_PATH = "../dataset/test"   # OR val folder
MODEL_PATH = "crop_disease_model.h5"

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# -------------------------
# LOAD CLASS LABELS
# -------------------------
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# -------------------------
# LOAD TEST DATA
# -------------------------
datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -------------------------
# PREDICTIONS
# -------------------------
y_true = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Crop Disease Detection")
plt.tight_layout()
plt.show()

# -------------------------
# CLASSIFICATION REPORT
# -------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

# -------------------------
# PRECISION / RECALL / F1 BAR GRAPH
# -------------------------
precision = [report[c]["precision"] for c in class_names]
recall = [report[c]["recall"] for c in class_names]
f1 = [report[c]["f1-score"] for c in class_names]

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1, width, label="F1-score")

plt.xticks(x, class_names, rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Precision, Recall & F1-score per Class")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
