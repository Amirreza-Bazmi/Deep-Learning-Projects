# LSTM Pooling Comparison

Welcome to my Deep Learning repository! 🚀

---

## 📂 Files in the Repository

1. **`LSTM.ipynb`**: Jupyter Notebook implementing the LSTM models with variations:
   - LSTM + Flatten layer.
   - LSTM + Global Max Pooling layer.
   - LSTM + Global Average Pooling layer.
2. **`README`**

---

## 📊 Overview of the Experiment

### Objective
To analyze the impact of adding a Flatten layer or Pooling layers (Max and Average) after an LSTM layer on the performance of the model for IMDB sentiment classification.

### Dataset
- **IMDB Dataset**: A binary sentiment classification dataset containing movie reviews labeled as positive or negative.

### Methodology
1. Train a baseline LSTM model without any additional layers after the LSTM.
2. Add a **Flatten** layer after the LSTM and evaluate the model.
3. Add a **Global Max Pooling** layer after the LSTM and evaluate the model.
4. Add a **Global Average Pooling** layer after the LSTM and evaluate the model.

### Results Summary
- **Baseline Accuracy**: 77% (without additional layers).
- **LSTM + Flatten**:
  - Accuracy: 86%
  - Improved performance by adding a Flatten layer.
- **LSTM + Global Max Pooling**:
  - Accuracy: 87%
  - Performed slightly better than Flatten.
- **LSTM + Global Average Pooling**:
  - Accuracy: 86%
  - Slightly less effective than Max Pooling.

**Key Insight**:
- Adding a Flatten or Pooling layer improves model performance significantly.
- Among Pooling layers, Max Pooling slightly outperforms Average Pooling.

---

## 🚀 How to Run

### 1. Clone the Repository
To get started, clone this repository using the following command:
```bash
git clone https://github.com/Amirreza-Bazmi/Deep-Learning-Projects.git
cd Deep-Learning-Projects/LSTM-Pooling-Comparison
```

### 2. Run the Notebook
- Open `LSTM.ipynb`.
- Follow the steps to:
  1. Preprocess the IMDB dataset.
  2. Train and evaluate the models with different layers.
  3. Compare the results.

---

## 📈 Example Outputs

### Accuracy Comparison
| Model                          | Accuracy |
|--------------------------------|----------|
| Baseline (LSTM Only)           | 77%      |
| LSTM + Flatten                | 86%      |
| LSTM + Global Max Pooling     | 87%      |
| LSTM + Global Average Pooling | 86%      |

---

## 📬 Contact
For questions or suggestions, feel free to reach out:
- **Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
- **GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)

---

Happy Coding! 🚀
