# 🧠 LSTM Text Classification

Welcome to my Deep Learning repository! 🚀
This project implements an LSTM-based text classification model with detailed preprocessing and custom metrics.

---

## 📂 Files in the Repository

1. **`lstm_text.ipynb`**: Jupyter Notebook implementing:
   - Text preprocessing (tokenization, stopword removal, stemming).
   - LSTM model with embedding and dense layers.
   - Custom metrics (precision and recall) for performance evaluation.
2. **`README.md`**: This file, providing an overview of the project.

---

## 📊 Overview of the Experiment

### Objective
To build and evaluate an LSTM-based model for text classification using a labeled dataset. The pipeline includes preprocessing steps and the use of custom metrics for a more detailed performance assessment.

### Dataset
- **FA-KES.csv**: A text dataset for classification tasks. (Ensure the dataset path is updated correctly in the notebook).

### Methodology
1. Preprocess text data:
   - Tokenization and sequence padding.
   - Stopword removal and stemming using NLTK.
2. Build and train an LSTM model with:
   - Embedding layer for word representation.
   - LSTM layer for capturing sequential patterns.
   - Dense layers for classification.
3. Evaluate the model using accuracy, precision, and recall metrics.

---

## 🚀 How to Run

### 1. Clone the Repository
To get started, clone the repository using:
```bash
git clone https://github.com/Amirreza-Bazmi/Deep-Learning-Projects.git
cd Deep-Learning-Projects/LSTM Text Classification
