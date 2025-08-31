# ✍️ Natural Language Inference (NLI) for Persian using ParsBERT

Welcome to this advanced Natural Language Processing (NLP) project! 🚀 This repository contains the implementation of a model for **Natural Language Inference (NLI)** on the **FarsTail** dataset. The project leverages the state-of-the-art **ParsBERT** model, fine-tuning it to classify the logical relationship between two Persian sentences.

---

## 📊 Project Overview

### **Objective**
The primary goal is to build and fine-tune a transformer-based model to perform textual entailment for the Persian language. The model takes a **premise** sentence and a **hypothesis** sentence and determines if the premise entails, contradicts, or is neutral with respect to the hypothesis.

### **Dataset**
* **FarsTail**: A Persian dataset specifically designed for the NLI task. The data is provided in separate files for training, validation, and testing (`Train-word.csv`, `Val-word.csv`, `Test-word.csv`).
* **Preprocessing**: The raw text data was cleaned and preprocessed (e.g., removing inconsistencies, special characters) to create the `cleaned_*.csv` files used for training.
* **Classes**: The task is a 3-class classification problem:
    1.  **Entailment**: The hypothesis logically follows from the premise.
    2.  **Contradiction**: The hypothesis contradicts the premise.
    3.  **Neutral**: There is no clear logical relationship.

### **Base Model**
* **ParsBERT (`HooshvareLab/bert-fa-base-uncased`)**: This project uses ParsBERT, a state-of-the-art monolingual BERT model pre-trained on a massive corpus of Persian text. It is designed to understand the nuances and context of the Persian language, making it an ideal foundation for this task.

---

## ⚙️ Methodology

The project follows a standard transfer learning pipeline for NLI tasks:

1.  **Data Preprocessing & Tokenization**:
    * The cleaned premise and hypothesis sentences were loaded from the `cleaned_*.csv` files.
    * The **`BertTokenizerFast`** for ParsBERT was used to format the sentence pairs into the required `[CLS] premise [SEP] hypothesis [SEP]` structure and convert them into token IDs.
2.  **Fine-Tuning Architecture**:
    * The pre-trained **`TFBertModel`** was loaded from the Hugging Face library.
    * A custom **classification head** was added on top of the base model's `[CLS]` token output. This head consists of a `Dense` layer with 3 units and a `softmax` activation function for multi-class classification.
3.  **Training**:
    * The model was compiled with the **`Adam`** optimizer, using a low learning rate suitable for fine-tuning.
    * The loss function used was **`CategoricalCrossentropy`** (or `SparseCategoricalCrossentropy`), appropriate for this multi-class task.
4.  **Evaluation**:
    * The model's performance was measured on a held-out test set using standard metrics like **Accuracy**, a detailed **Classification Report** (Precision, Recall, F1-Score), and a **Confusion Matrix**.

---

## 🛠️ Technologies Used

* **Frameworks**: Pytorch
* **Libraries**: Hugging Face Transformers, Pandas, Scikit-learn

---

## 📈 Results

The fine-tuned ParsBERT model demonstrated strong performance in understanding the logical relationships between Persian sentences.

* **Final Test Accuracy (for 3 epochs)**: **[94.75]%**
* **Final Test F1-Score (for 3 epochs)**: **[0.9292]**

**(It is highly recommended to add a screenshot of your Classification Report or Confusion Matrix plot here)**

---

## 🚀 How to Run

### **1. Clone the Repository**
```bash
git clone https://github.com/Amirreza-Bazmi/Deep-Learning-Projects.git
cd Deep-Learning-Projects/Natural Language Inference (NLI) for Persian using ParsBERT
```

---

## 📬 Contact

For any questions, suggestions, or collaboration opportunities, feel free to contact me:

- **📧 Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
- **🐙 GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)
