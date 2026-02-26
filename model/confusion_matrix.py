import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# CONFIGURATION
# -------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 16
TEST_PATH = "../dataset/test"

# -------------------------
# LOAD TRAINED MOBILENET MODEL
# -------------------------
model = load_model("crop_disease_model.h5")
print("✅ MobileNet model loaded")

# -------------------------
# LOAD CLASS LABELS
# -------------------------
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

# Convert index → class name
class_labels = {v: k for k, v in class_indices.items()}
class_names = [class_labels[i] for i in range(len(class_labels))]

print("Classes:", class_names)

# -------------------------
# LOAD TEST DATA
# -------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -------------------------
# MODEL PREDICTION
# -------------------------
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_true, y_pred)

print("\n📊 Confusion Matrix:\n")
print(cm)

# -------------------------
# CLASSIFICATION REPORT
# -------------------------
print("\n📄 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -------------------------
# PLOT CONFUSION MATRIX
# -------------------------
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
plt.title("Confusion Matrix - MobileNet Crop Disease Detection")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

# Add numbers inside matrix cells
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
