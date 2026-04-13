# 🩺 Pneumonia Classifier (Deep Learning + Explainable AI)

## 📌 Overview

This project implements a **deep learning-based pneumonia detection system** using chest X-ray images.

It uses:

* **EfficientNet-B0** for classification
* **PyTorch** for training
* **Grad-CAM** for explainability

The system classifies X-rays into:

* **NORMAL**
* **PNEUMONIA**

This project demonstrates how AI can assist in **medical diagnosis** while providing **visual explanations** of model decisions.

---

## 🧠 Model Architecture

* Backbone: **EfficientNet-B0 (pretrained)**
* Custom head:

  * Dropout (regularization)
  * Fully connected layer (2 classes)

Uses **transfer learning** to achieve strong performance on medical imaging data.

---

## 🛠️ Features

### ✅ Classification

* Binary classification: NORMAL vs PNEUMONIA
* Data augmentation applied
* Optimized using Adam optimizer

### 🔍 Explainable AI (Grad-CAM)

* Generates heatmaps showing model attention
* Highlights important regions in X-rays
* Improves interpretability

### 📊 Evaluation

* Accuracy
* F1-score
* Confusion matrix
* Classification report

---

## 📈 Results

* **Test Accuracy:** ~82–88%
* **F1 Score:** ~0.87–0.91

---

## 🖼️ Grad-CAM Visualizations

These images show **where the model focuses** when making predictions.

* 🔴 Red = high importance
* 🔵 Blue = low importance

---

### Pneumonia Cases

#### Image 1


<img width="470" height="470" alt="IM-0122-0001_gradcam" src="https://github.com/user-attachments/assets/60b11aa8-f34b-4571-8b1c-f382023fc3d8" />




➡️ Strong activation in a localized lung region, consistent with pneumonia infection.


---

#### Image 2


<img width="470" height="470" alt="person1_bacteria_1_gradcam" src="https://github.com/user-attachments/assets/71f577b5-ac41-4de4-81a5-6183dfd62e75" />




➡️ Activation concentrated in the lower lung, typical of bacterial pneumonia.


---

#### Image 3


<img width="470" height="470" alt="person2_bacteria_3_gradcam" src="https://github.com/user-attachments/assets/1090625e-9027-4de9-9cd6-5f63e78258e5" />




➡️ Focused attention on one side of the lung, indicating infection.


---

### Viral / Diffuse Pattern

#### Image 4


<img width="470" height="470" alt="person124_virus_247_gradcam" src="https://github.com/user-attachments/assets/d9a07ea2-eb36-4f59-a16c-4998e0afa02c" />




➡️ Diffuse activation across lung regions, often associated with viral pneumonia.


---

### Normal Cases

#### Image 5


<img width="470" height="470" alt="IM-0140-0001_gradcam" src="https://github.com/user-attachments/assets/4a74c715-0f4e-40d6-8c85-ad79556bc2ac" />




➡️ No strong localized hotspots, indicating normal lung structure.


---

#### Image 6


<img width="470" height="470" alt="IM-0210-0001_gradcam" src="https://github.com/user-attachments/assets/d8d97da0-7132-4ab6-9022-8b8534f5a6ea" />




➡️ More uniform activation, suggesting absence of pneumonia.


---

## 📥 Dataset Setup

Dataset:
👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

After downloading, organise it like:

```text
data/
├── train/
├── val/
└── test/
```

Update dataset path in:

```text
utils/config.py
```

---

## 🚀 How to Run

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-classifier.git
cd pneumonia-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train model

```bash
python main.py --mode train
```

### 4. Evaluate model

```bash
python main.py --mode test
```

### 5. Generate Grad-CAM

```bash
python main.py --mode gradcam --image path_to_image.jpg
```

---

## 📂 Project Structure

```text
pneumonia_classifier/
├── models/
├── utils/
├── images/        ← place Grad-CAM images here
├── main.py
├── requirements.txt
└── README.md
```

---

## 🎯 Purpose

This project demonstrates:

* End-to-end deep learning pipeline
* Medical image classification
* Explainable AI integration

It provides a foundation for building and validating AI systems before applying them in real-world healthcare scenarios.

---

## 📚 Technologies Used

* Python
* PyTorch
* torchvision
* timm
* scikit-learn
* matplotlib

---

## 👨‍💻 Author



---

## 📜 License

MIT License
