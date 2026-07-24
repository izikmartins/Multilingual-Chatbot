# Movie Recommendation System with Multilingual Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)

A hybrid movie recommendation system with a conversational multilingual interface supporting **English**, **Yoruba**, and **Igbo**. The system combines **content-based filtering** (TF-IDF + Cosine Similarity) and **collaborative filtering** (Singular Value Decomposition - SVD) with a **feedforward deep neural network** for intent classification.

---

## 📋 Overview

E-commerce platforms struggle with information overload. This system helps users discover movies through natural conversation. It provides:

- **Personalised recommendations** using a hybrid SVD + content-based cascade architecture.
- **Multilingual chatbot** that understands and responds in English, Yoruba, and Igbo.
- Validated on a **modified MovieLens 100k dataset** augmented with Nollywood movies.

> **📄 Preprint**: This work is published as *"Movie Recommendation System for E-Commerce Using a Multilingual Chatbot"*.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Multilingual NLP** | Intent classification in 3 languages (44 intents, 332 patterns). |
| **Hybrid Recommender** | Content‑based candidate generation + SVD rating prediction. |
| **Graphical Interface** | Tkinter GUI with voice input (speech-to-text) and text-to-speech. |
| **Real‑time Ranking** | Top 6 recommendations ranked by predicted user rating. |
| **Cold‑Start Ready** | New users can get recommendations by entering a movie title. |

---


---

## 🛠️ Installation

### 1. Prerequisites
- Python 3.8 or higher
- pip

### 2. Clone the Repository
```bash
git clone https://github.com/izikmartins/Multilingual-Chatbot-DataSet.git
cd Multilingual-Chatbot-DataSet

### 3.  Install Dependencies
pip install -r requirements.txt

### 4. Download NLTK Data (Required)
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

📊 Dataset
The system uses a modified version of the MovieLens 100k dataset.

Dataset	Original	Modified	Final (Cleaned)
Movies	9,742	+165 Nollywood	9,902 unique
Users	610	+10 synthetic	620 unique
Ratings	100,836	+192 synthetic	101,029
The dataset includes unique genres such as Nollywood, Yoruba, and Igbo.

🤖 Trained Model Configuration
Chatbot Model (chatbot_model.h5)
Type: Feedforward Neural Network (Multilayer Perceptron)

Input: Bag-of-Words (BoW) vector of length 384

Hidden Layer 1: 128 neurons, ReLU, Dropout 0.5

Hidden Layer 2: 64 neurons, ReLU, Dropout 0.5

Output Layer: 44 neurons, Softmax

Optimizer: SGD (lr=0.01, momentum=0.9, Nesterov)

Loss: Categorical Crossentropy

Epochs: 200

Batch Size: 5

Classification Threshold: 0.25

SVD Recommender (svd_model.pkl)
Library: Surprise

Latent Factors: 50

Epochs: 20

Learning Rate: 0.005

Regularization: 0.02

Random Seed: 42

🚀 Usage
Run the GUI Application:
python src/chatbotgui.py

Get Recommendations:

- User ID (1-620): Returns top 6 movies with predicted ratings.
- Movie Title: Returns top 9 similar movies (content-based).

Retrain Models from Scratch
Train the Chatbot:
bash:
python src/train_chatbot.py
Outputs: models/chatbot_model.h5, models/words.pkl, models/classes.pkl

Train the SVD Recommender:
bash
python src/hybrid.py
Outputs: models/svd_model.pkl

Reproduce Results (Evaluation)
Run the complete evaluation suite to generate all metrics:
bash
python src/evaluate.py
This script outputs:
- Chatbot Classification: Accuracy, Precision, Recall, F1-score
- SVD Rating Prediction: RMSE, MAE (5‑fold CV)
- Content-Based Hit/Fault: Genre‑matching accuracy (K=9)

📈 Evaluation Results
The following results are reported in the manuscript and are fully reproducible using evaluate.py.

Component	Metric	Value
Chatbot	Accuracy	87.9%
Chatbot	Precision (Weighted)	87.5%
Chatbot	Recall (Weighted)	87.9%
Chatbot	F1-Score (Weighted)	87.6%
SVD	RMSE	0.8794
SVD	MAE	0.6761
Content‑Based	Hit Rate (K=9)	92.93%
Content‑Based	Fault Rate (K=9)	7.07%

🙏 Acknowledgments
MovieLens for the base dataset.

Covenant University (CUCRID) for sponsorship and support.

Supervisors and colleagues for their guidance.


---

### Optional: `requirements.txt` (if not already present)
To ensure your repository is fully functional, ensure your `requirements.txt` looks like this:
nltk>=3.6.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
scikit-surprise>=1.1.0
tensorflow>=2.6.0
keras>=2.6.0
customtkinter>=5.0.0
pillow>=8.3.0
speechrecognition>=3.8.0
pyttsx3>=2.90
playsound>=1.2.2
