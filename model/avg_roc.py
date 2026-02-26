import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# -------------------------
# CONFIGURATION
# -------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 16
TEST_PATH = "../dataset/test"   # or validation folder
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
num_classes = len(class_names)

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
# TRUE LABELS & PREDICTIONS
# -------------------------
y_true = test_data.classes
y_pred_prob = model.predict(test_data)

y_true_bin = label_binarize(y_true, classes=range(num_classes))

# -------------------------
# COMPUTE ROC FOR EACH CLASS
# -------------------------
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# -------------------------
# MACRO-AVERAGE ROC
# -------------------------
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= num_classes

macro_auc = auc(all_fpr, mean_tpr)

# -------------------------
# PLOT MACRO-AVERAGE ROC
# -------------------------
plt.figure(figsize=(8, 7))

plt.plot(
    all_fpr, mean_tpr,
    color="blue", lw=3,
    label=f"Macro-average ROC (AUC = {macro_auc:.2f})"
)

plt.plot([0, 1], [0, 1], "k--", lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Macro-Average ROC Curve – Crop Disease Detection")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
