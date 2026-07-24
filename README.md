# 👟 Real vs AI-Generated Shoe Image Detection

A deep learning-based binary image classification system that distinguishes **real shoe images** from **AI-generated shoe images** using **EfficientNetB3** and **TensorFlow**.

The project follows a modular software engineering approach with separate components for preprocessing, training, evaluation, prediction, and deployment through a Streamlit web application.

---

## 📌 Project Overview

With the rapid growth of generative AI, AI-generated product images have become increasingly realistic. This project aims to classify whether a shoe image is:

- ✅ Real
- 🤖 AI-generated

The model is built using **Transfer Learning** with **EfficientNetB3**, fine-tuned on a custom dataset and deployed using **Streamlit** for real-time predictions.

---

## 🚀 Features

- Binary image classification (Real vs AI-generated)
- Transfer Learning using EfficientNetB3
- Image preprocessing and normalization pipeline
- Data augmentation for improved generalization
- Fine-tuning of pretrained layers
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - Confusion Matrix
- Single image prediction script
- Interactive Streamlit web application
- Modular project structure

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Model | EfficientNetB3 |
| Image Processing | Pillow, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning Metrics | Scikit-learn |
| Web Application | Streamlit |

---

## 📂 Project Structure

```text
real-vs-ai-generated-shoe-image-detection/
│
├── app/
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── dataset/
│   └── test_images/
│
├── models/
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── confusion_matrix.py
│   ├── roc_curve.py
│   └── predict.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/sagarucareer/real-vs-ai-generated-shoe-image-detection.git

cd real-vs-ai-generated-shoe-image-detection
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

Train the model

```bash
python -m src.train
```

The best and final models will be saved inside:

```text
models/
```

---

## 📊 Model Evaluation

Run evaluation metrics

```bash
python -m src.evaluate
```

Generate Confusion Matrix

```bash
python -m src.confusion_matrix
```

Generate ROC Curve

```bash
python -m src.roc_curve
```

---

## 🔍 Predict on a Single Image

```bash
python -m src.predict
```

Enter the image path when prompted.

---

## 🌐 Run the Streamlit App

```bash
streamlit run app/app.py
```

Upload a shoe image and the application will display:

- Predicted class
- Confidence score
- Visual result

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **70.89%** |
| Precision | **62.72%** |
| Recall | **80.90%** |
| F1 Score | **70.66%** |
| ROC-AUC | **79.60%** |

---

## 💡 Future Improvements

- Train on a larger and more diverse dataset
- Experiment with newer architectures (ConvNeXt, EfficientNetV2)
- Hyperparameter optimization
- Deploy the application on Streamlit Cloud or Hugging Face Spaces
- Add Grad-CAM visualization for model explainability

---

## 👨‍💻 Author

Developed as part of a deep learning project to explore transfer learning, image classification, and deployment of computer vision models using TensorFlow and Streamlit.