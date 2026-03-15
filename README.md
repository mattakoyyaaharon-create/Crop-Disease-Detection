Crop Disease Detection Using MobileNet (Transfer Learning)

Batch ID: Batch-07
Course: Undergrad Major Project 2026
Institution: PACE Institute of Technology & Sciences

👥 Team Members
Name	Roll Number	GitHub Handle
KATTEDA SASI KUMAR	22KQ1A6144	@yourgithub
MADDISETTY DIVYA SAI CHARAN	22KQ1A6149	@yourgithub
ANNAVARAPU VANAJA	22KQ1A6102	@yourgithub
KODUMURI VIJAYA VASANTHI LAKSHMI	22KQ1A6111	@yourgithub
THUMMALA UMA VAISHNAVI	22KQ1A6127	@yourgithub
🚀 Project Overview
📌 Problem Statement

Crop diseases significantly affect agricultural productivity and food security. Traditional manual disease detection methods are time-consuming, error-prone, and not scalable for large farms. There is a need for an automated, accurate, and real-time disease detection system suitable for deployment in rural agricultural environments.

🎯 Key Objective

To develop a lightweight MobileNet-based transfer learning model capable of achieving 92–95% test accuracy for real-time tomato leaf disease classification.

📊 Dataset Information

Source: Kaggle – Tomato Leaf Disease Dataset
https://www.kaggle.com/code/samanfatima7/tomato-leaf-disease-94-accuracy/input

Description:

Multi-class tomato leaf disease images

Categories:

Tomato Early Blight

Tomato Late Blight

Tomato Leaf Mold

Tomato Target Spot

Tomato Mosaic Virus

Healthy

Preprocessing Steps:

Image resizing to 224 × 224

Pixel normalization (rescale = 1./255)

Data augmentation:

Rotation (±30°)

Zooming (30%)

Horizontal flipping

Width & height shifting

Train/Validation/Test split (80/10/10)

⚠️ Dataset is excluded from this repository due to size. Please download directly from Kaggle.

🧠 Model Architecture & Methodology
🔹 Algorithm / Model

MobileNet (Transfer Learning)

Depthwise Separable Convolution

Fine-tuning of higher layers

🔹 Framework

TensorFlow

Keras

Flask (for deployment)

HTML/CSS/JavaScript (Frontend)

🔹 Training Strategy

Pre-trained ImageNet weights

Frozen base layers initially

Added:

Global Average Pooling

Dense Layer (ReLU)

Dropout (Regularization)

Softmax Output Layer

Fine-tuning for improved generalization

📈 Results & Performance

The proposed MobileNet model achieved strong performance across all evaluation metrics.

Metric	Value
Training Accuracy	97%
Validation Accuracy	94%
Test Accuracy	92–95%
Precision	0.93
Recall	0.93
F1-Score	0.95
AUC	0.95
🔹 Performance Highlights

Smooth convergence

Minimal overfitting

Fast inference time (< 2 seconds)

Strong ROC curve (AUC > 0.90 for all classes)

🌐 Application Deployment

The model is deployed using:

Backend: Flask

Frontend: HTML, CSS, JavaScript

Model File: .h5

Class Labels: .json

🔹 Features

Upload leaf image

Capture via camera

Real-time prediction

Confidence score display

User-friendly interface for farmers

🛠️ Installation & Usage
1️⃣ Requirements

Ensure Python 3.10+ is installed.

Install required libraries:

pip install -r requirements.txt
2️⃣ Run the Application
python app.py

Open browser:

http://127.0.0.1:5000/
📂 Project Structure
📁 crop-disease-detection
│
├── app.py
├── model.h5
├── labels.json
├── requirements.txt
├── README.md
│
├── 📁 static
├── 📁 templates
├── 📁 uploads
└── 📁 docs
    ├── accuracy_plot.png
    ├── confusion_matrix.png
    └── roc_curve.png
🔮 Future Scope

Extend to multiple crop types

Deploy as Android mobile app

Integrate regional language support

Cloud-based scalable deployment

📌 Conclusion

The proposed MobileNet-based transfer learning model provides an efficient, accurate, and lightweight solution for real-time crop disease detection. Its strong performance metrics and low computational requirements make it suitable for practical agricultural deployment.