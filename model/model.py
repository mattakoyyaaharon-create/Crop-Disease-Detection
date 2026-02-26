import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json


# -------------------------
# CONFIGURATION
# -------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 15
EPOCHS_FINE = 10

TRAIN_PATH = "../dataset/train"
VAL_PATH   = "../dataset/val"
TEST_PATH  = "../dataset/test"


# -------------------------
# DATA GENERATORS
# -------------------------

# Training (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Validation & Test (NO augmentation ❗)
test_val_datagen = ImageDataGenerator(rescale=1./255)


# -------------------------
# LOAD DATA
# -------------------------
train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = test_val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_data = test_val_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
print("Class indices:", train_data.class_indices)


# -------------------------
# LOAD MOBILENET (TRANSFER LEARNING)
# -------------------------
base_model = MobileNet(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze base model
base_model.trainable = False


# -------------------------
# CUSTOM CLASSIFIER HEAD
# -------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

#Combines MobileNet + custom head into one model

model = Model(inputs=base_model.input, outputs=output)


# -------------------------
# STAGE 1: TRAIN CLASSIFIER HEAD
# -------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🔹 Training classifier head...")
history_head = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD
)


# -------------------------
# STAGE 2: FINE-TUNING (NO LEAKAGE)
# -------------------------
print("\n🔹 Fine-tuning top MobileNet layers...")

# Unfreeze only last 15 layers
for layer in base_model.layers[-15:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE
)


# -------------------------
# FINAL TEST EVALUATION (UNSEEN DATA)
# Evaluates model on completely unseen data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"\n✅ Final Test Accuracy (No Leakage): {test_accuracy * 100:.2f}%")


# -------------------------
# SAVE MODEL & LABELS
# -------------------------
model.save("crop_disease_model.h5")

with open("class_labels.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=4)

print("✅ Model and labels saved")


# -------------------------
# ACCURACY & LOSS PLOTS
# -------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history_head.history["accuracy"], label="Head Train")
plt.plot(history_head.history["val_accuracy"], label="Validation")
plt.plot(history_fine.history["accuracy"], label="Fine Tune")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_head.history["loss"], label="Head Train")
plt.plot(history_head.history["val_loss"], label="Validation")
plt.plot(history_fine.history["loss"], label="Fine Tune")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()
