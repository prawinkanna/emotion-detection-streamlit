# 😊 Emotion Detection App (CNN + Streamlit)

A **Level-4 Deep Learning project** that detects **human emotions from facial images** using a **Convolutional Neural Network (CNN)** and is deployed as a **public Streamlit web application**.

🌐 **Live Application**  
👉 https://emotion-detection-app-dbtyqknfu6ye8b7ene68j4.streamlit.app/

---

## 🚀 Project Overview

This application allows users to **upload a face image** and predicts the **most likely human emotion** along with a **confidence score** and a **probability distribution chart**.

The system uses a **pretrained CNN model** (FER-2013 based) and performs **real-time inference** without exposing any training data.

---

## 🧠 Emotions Detected

The model predicts one of the following emotions:

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- Convolutional Neural Network (CNN)  
- OpenCV  
- Streamlit  
- NumPy  
- Pillow  

---

## 📂 Project Structure

emotion-detection-streamlit/
├── app.py
├── emotion_model.h5
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy code

---

## ▶️ How the Application Works

1. User uploads an image  
2. Image is converted to grayscale  
3. Image is resized to **64 × 64**  
4. Pixel values are normalized  
5. CNN model predicts emotion probabilities  
6. Highest probability emotion is displayed  

---

## 📊 Understanding the Output Graph (IMPORTANT)

After prediction, the app displays a **bar chart**.

### 🔹 X-Axis (Horizontal Axis)
Represents **emotion classes**:

Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral

yaml
Copy code

Each label corresponds to one possible emotion.

---

### 🔹 Y-Axis (Vertical Axis)
Represents the **probability / confidence** of each emotion.

- Range: **0.0 → 1.0**
- All values **sum to 1** (Softmax output)

---

### 📈 Example Probability Diagram

Probability
1.0 ┤
0.8 ┤
0.6 ┤ █
0.4 ┤ █ █
0.2 ┤ █ █ █
0.0 ┼──────────────────────────
Angry Happy Sad Neutral

yaml
Copy code

✔ Highest bar = **Predicted emotion**  
✔ Lower bars = Less likely emotions  

---

## 🧠 How Confidence is Calculated

The CNN uses a **Softmax activation** in the final layer.

Example output:
[0.12, 0.05, 0.09, 0.18, 0.37, 0.11, 0.08]

yaml
Copy code

- Highest value → prediction  
- Example: `0.37 → Sad (37% confidence)`

---

## ⚠️ Important Note on Confidence

A **low confidence value** does NOT mean the app is wrong.

It indicates:
- Mixed facial expressions
- Lighting variations
- Real-world ambiguity

This reflects **honest AI behavior**, not overfitting.

---

## ⚠️ Disclaimer

This project is developed **strictly for educational and demonstration purposes**.  
Predictions may not be 100% accurate and should not be used for medical or psychological diagnosis.

---

## 🏆 Learning Outcomes

- Deep understanding of CNN architecture  
- Transfer learning concepts  
- Image preprocessing for deep learning  
- Model deployment using Streamlit  
- Real-world debugging and interpretation of ML outputs  

---

## 📌 Resume / Interview Statement

> “I built and deployed a Level-4 CNN-based emotion detection web application using Streamlit and TensorFlow, including probability-based output visualization.”

---

## 🌐 Live Demo

👉 https://emotion-detection-app-dbtyqknfu6ye8b7ene68j4.streamlit.app/
