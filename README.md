# 🩺 Skin Disease Detection using Deep Learning

An AI-powered web application that classifies skin diseases from images using **EfficientNetB3** and provides visual explanations via **Grad-CAM**.

---

## 🚀 Live Features

* 📤 Upload skin image for instant prediction
* 🧠 Deep Learning model (EfficientNetB3)
* 📊 Confidence score for predictions
* 🔥 Grad-CAM visualization (model explainability)
* 🎯 Multi-class classification (7 diseases)
* 🌐 Clean, modern Streamlit UI

---

## 🧠 Problem Statement

Millions of people lack access to dermatologists, especially in rural areas.
This project aims to provide a **fast, accessible AI-based skin disease detection system**.

---

## 🏗️ Tech Stack

* **Language:** Python
* **ML/DL:** TensorFlow, Keras
* **Model:** EfficientNetB3 (Transfer Learning)
* **Frontend:** Streamlit
* **Image Processing:** OpenCV, PIL
* **Data Handling:** Pandas, NumPy

---

## 📊 Dataset

* **HAM10000 Dataset** (Human Against Machine)
* ~10,000 dermatoscopic images
* 7 classes:

  * Melanoma (mel)
  * Melanocytic Nevi (nv)
  * Basal Cell Carcinoma (bcc)
  * Actinic Keratosis (akiec)
  * Benign Keratosis (bkl)
  * Dermatofibroma (df)
  * Vascular Lesions (vasc)

---

## 🧠 Model Architecture

* Base: EfficientNetB3 (pretrained on ImageNet)
* Head:

  * Global Average Pooling
  * Dense (512) + Dropout
  * Dense (256) + Dropout
  * Softmax (7 classes)

---

## ⚙️ How It Works

1. User uploads skin image
2. Image is resized to 224×224 and normalized
3. Model predicts probabilities for 7 classes
4. Top prediction is displayed with confidence
5. Grad-CAM highlights important regions

---

## 📂 Project Structure

```
skin-disease-detector/
│
├── app.py
├── split_dataset.py
├── eda_preprocessing.py
├── requirements.txt
├── run_all.bat
├── .gitignore
│
├── models/
├── utils/
│
├── data/        # (not included)
├── outputs/     # (not included)
```

---

## ⚡ Quick Start (Recommended)

### 1️⃣ Clone Repository

```
git clone https://github.com/YOUR_USERNAME/skin-disease-detector.git
cd skin-disease-detector
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Download Pretrained Model

👉 Download model from:
**[Add your Google Drive link here]**

Place it in:

```
outputs/models/final_model.h5
```

---

### 4️⃣ Run Application

```
streamlit run app.py
```

---

## 🔄 Full Training (Optional)

If you want to train from scratch:

### 1. Download Dataset (Kaggle)

Place in:

```
data/raw/
```

---

### 2. Split Dataset

```
python split_dataset.py
```

---

### 3. Train Model

```
python -m models.train_model
```

---

## 📈 Model Performance

* Accuracy: ~67% (baseline)
* Optimized: ~80–88% (with class weights + fine-tuning)

---

## ⚠️ Limitations

* Model is trained on **only 7 classes**
* Cannot detect unknown diseases (e.g., Becker’s Nevus)
* Predictions are based on **visual similarity, not diagnosis**

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**.
> It is **not a medical diagnosis tool**.
> Always consult a certified dermatologist.

---

## 💡 Future Improvements

* Add more datasets (ISIC, DermNet)
* Unknown disease detection
* Mobile app version
* Cloud deployment
* Better class balancing

---

## 👨‍💻 Author

**Meet Daslaniya**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
