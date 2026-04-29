# Handwriting Recognition System

A hybrid handwriting verification and recognition system developed using Deep Learning, Feature Fusion, and Computer Vision techniques. The system compares two handwriting samples and determines whether they belong to the same writer.

---

## 📌 Project Overview

This project combines:

* Siamese Convolutional Neural Network (CNN)
* Handcrafted Feature Extraction
* Feature Fusion Techniques
* Random Forest Classification
* Streamlit Web Application

The system analyzes handwriting images, extracts deep and handcrafted features, computes similarity scores, and predicts whether the handwriting samples are from the same writer.

---

## 🚀 Features

* Upload two handwriting images
* Automatic image preprocessing
* CNN-based similarity computation
* ORB and handcrafted feature extraction
* Feature fusion for improved accuracy
* Match / No Match prediction
* Similarity score display
* Streamlit-based interactive interface

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Scikit-learn
* NumPy
* Matplotlib
* Streamlit

---

## 🧠 Algorithms Used

### 1. Siamese CNN

Used for learning visual similarity between handwriting images.

### 2. Feature Fusion

Combines CNN similarity scores with handcrafted features.

### 3. ORB Feature Matching

Used for keypoint extraction and local feature comparison.

### 4. Random Forest Classifier

Used for final classification of handwriting pairs.

---

## 📂 Project Structure

```text
Handwriting-Recognition-System/
│
├── app.py
├── preprocessing.py
├── model.py
├── fusion.py
├── features.py
├── utils.py
├── requirements.txt
├── README.md
├── dataset/
├── outputs/
└── images/
```

---

## 📊 Dataset Used

The project was trained and evaluated using the **CVL Handwriting Dataset**, which contains handwriting samples from approximately 1000 writers.

Dataset Reference:

F. Kleber, S. Fiel, M. Diem, and R. Sablatnig,
“CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting,” ICDAR, 2013.

---

## 📈 Performance

* Accuracy: 75%
* AUC Score: 0.88
* High Precision with low False Acceptance Rate

---

## ▶️ How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/handwriting-recognition-system.git
```

### Step 2: Open Project Folder

```bash
cd handwriting-recognition-system
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit Application

```bash
streamlit run app.py
```

---

## 📷 Output

The system displays:

* Similarity Score
* Match / No Match Result
* CNN Similarity
* Fusion Probability
* ROC Curve
* Confusion Matrix

---

## 📚 References

1. E. Rublee et al., “ORB: An efficient alternative to SIFT or SURF,” ICCV, 2011.
2. F. Kleber et al., “CVL-Database,” ICDAR, 2013.
3. Y. LeCun et al., “Deep Learning,” Nature, 2015.
4. J. Bromley et al., “Signature verification using a Siamese neural network,” 1994.
5. OpenCV Documentation – [https://opencv.org](https://opencv.org)
6. Streamlit Documentation – [https://streamlit.io](https://streamlit.io)

---

## 👨‍💻 Developed For

Mini Project – Bachelor of Technology in Computer Science and Engineering.

---

## 📌 Future Improvements

* Real-time handwriting recognition
* Multi-language handwriting support
* Transformer-based architectures
* Mobile application deployment
* Larger dataset training

---
