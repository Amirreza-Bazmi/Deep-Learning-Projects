# 🎬 IMDB Movie Review Sentiment Analysis with RNN

Welcome to my Natural Language Processing (NLP) repository! 🚀 This project features a Recurrent Neural Network (RNN) built with Keras for binary sentiment classification of IMDB movie reviews. The model can determine whether a review is positive or negative.

---

## 📊 Project Overview

### Objective
The primary goal is to build and train a `SimpleRNN` model to understand and classify the sentiment of text data. This project showcases an end-to-end NLP workflow, from data loading and text preprocessing to model training and, most importantly, inference on custom, user-written sentences.

### Dataset
-   **IMDB Movie Review Dataset**: A classic dataset for binary sentiment analysis, containing 50,000 movie reviews pre-labeled as positive (1) or negative (0).
-   The vocabulary was limited to the top **10,000 most frequent words**.

### Methodology
1.  **Data Loading & Preprocessing**:
    -   The IMDB dataset was loaded directly from `keras.datasets`, already tokenized into integer sequences.
    -   **`pad_sequences`** was applied to ensure all reviews have a uniform length of **250 words**, by truncating longer reviews and padding shorter ones.
2.  **Model Architecture**:
    -   An **`Embedding` layer** was used as the first layer to convert integer-encoded words into dense 32-dimensional vectors. This allows the model to learn semantic similarities between words.
    -   A **`SimpleRNN` layer** with 32 units processes the sequence of word embeddings to capture contextual information.
    -   A final **`Dense` layer** with a single neuron and a sigmoid activation (implicit in the loss function) outputs the sentiment prediction (close to 1 for positive, 0 for negative).
3.  **Training & Evaluation**:
    -   The model was compiled using `binary_crossentropy` loss, suitable for binary classification, and the `rmsprop` optimizer.
    -   It was trained for 5 epochs, achieving a peak validation accuracy of approximately **85%**.
4.  **Custom Inference Engine**:
    -   A key feature of this project is a script that takes a new, custom sentence, preprocesses it using the same vocabulary and padding, and feeds it to the trained model to predict its sentiment in real-time.

---

## 🛠️ Technologies Used

-   **Framework**: TensorFlow, Keras
-   **Libraries**: NumPy

---

## 📈 Results

-   The model achieved a **peak validation accuracy of ~85%** on the test set.
-   The inference engine successfully classifies the sentiment of custom sentences.

**Example Prediction:**
-   **Input**: *"this movie is bad i hated it and wouldn't watch it again was one of the worst things i've ever watched"*
-   **Prediction**: **Negative** (Model output: `[[-0.236]]`, which is < 0)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/Amirreza-Bazmi/Your-Repo-Name.git](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects.git)
cd Deep-Learning-Projects/IMDB Movie Review Sentiment Analysis with RNN
```

---

## 📬 Contact

For any questions or collaboration opportunities, feel free to get in touch:

-   **📧 Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
-   **🐙 GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)
