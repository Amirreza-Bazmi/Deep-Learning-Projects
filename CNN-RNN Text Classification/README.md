# 🧠 CNN-RNN Text Classification

Welcome to the **CNN-RNN Text Classification** project! 🚀 This repository combines the power of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to classify text data efficiently.

---

## 📂 Files in the Repository

1. **`cnn_rnn_classification.ipynb`**: Jupyter Notebook implementing:
   - Text preprocessing (tokenization, stopword removal, stemming).
   - A hybrid CNN-RNN model with embedding, convolutional, pooling, and LSTM layers.
   - Custom metrics (precision and recall) for performance evaluation.
2. **`README.md`**: This file, providing an overview of the project.

---

## 📊 Overview of the Experiment

### Objective
To develop a hybrid CNN-RNN model for text classification, leveraging the feature extraction capabilities of CNN and the sequence learning power of RNN (LSTM).

### Dataset
- **FA-KES.csv**: A labeled text dataset (e.g., fake news detection). Ensure the dataset path is updated correctly in the notebook.

### Methodology
1. Preprocess text data:
   - Tokenization and sequence padding.
   - Stopword removal and stemming using NLTK.
2. Build and train a CNN-RNN hybrid model:
   - **Embedding Layer**: Word embeddings for representing text data.
   - **Convolutional Layer**: Feature extraction via 1D convolutions.
   - **Max Pooling Layer**: Down-sampling to reduce dimensionality.
   - **LSTM Layer**: Sequential learning for text data.
   - **Dense Layer**: For final classification.
3. Evaluate the model:
   - Metrics: Accuracy, precision, recall.

---

## 🚀 How to Run

### 1. Clone the Repository
Clone this repository using the following command:
```bash
git clone https://github.com/Amirreza-Bazmi/Deep-Learning-Projects.git
cd Deep-Learning-Projects/CNN-RNN Text Classification
```

---

## 📬 Contact

For any questions, suggestions, or collaboration opportunities, feel free to contact me:

- **📧 Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
- **🐙 GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)
