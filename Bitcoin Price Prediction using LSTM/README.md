# 📈 Bitcoin Price Prediction using LSTM

Welcome to this time-series forecasting project! 🚀 This repository contains a sophisticated Long Short-Term Memory (LSTM) network built with TensorFlow and Keras to predict the closing price of Bitcoin using high-frequency historical data.

---

## 📊 Project Overview

### Objective
The goal of this project is to develop a robust deep learning model for time-series prediction. It demonstrates an end-to-end pipeline for forecasting financial data, including data preprocessing, sequence generation, building a stacked LSTM model, and evaluating its performance with key regression metrics.

### Dataset
-   **`btc_price_data_2018_2024.csv`**: A high-frequency dataset containing Bitcoin price data at **1-day intervals** from 2018 to 2024. The model is trained on the "Close" price.

### Methodology
1.  **Data Preprocessing**:
    -   The "Close" price data was extracted and normalized using `MinMaxScaler` to a range of `[0, 1]`, which is crucial for the stability and performance of LSTM networks.
2.  **Sequence Generation**:
    -   The time-series data was transformed into supervised learning sequences using a sliding window approach. A window of the past **30 time steps** (30 minutes) was used as input features to predict the next time step's price.
3.  **Model Architecture**:
    -   A **Stacked LSTM Network** was implemented with two LSTM layers.
    -   `return_sequences=True` was used on the first LSTM layer to ensure its output sequence could be fed into the second layer.
    -   `Dropout` layers were included to prevent overfitting.
    -   A final `Dense` layer with a single neuron outputs the predicted price.
4.  **Training & Optimization**:
    -   The model was trained using the `Adam` optimizer and `Mean Squared Error` loss function.
    -   An **`EarlyStopping` callback** was implemented to monitor validation loss, preventing overfitting and ensuring the model retains its best-performing weights.
5.  **Evaluation**:
    -   Predictions were made on both the training and test sets.
    -   The scaled predictions were converted back to their original price scale using `inverse_transform` before evaluation.
    -   Performance was measured using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

---

## 🛠️ Technologies Used

-   **Framework**: TensorFlow, Keras
-   **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib

---

## 📈 Results


### Visualization of Predictions
The plots below show the model's predictions against the actual prices for the training and test datasets.

**(Here you can add screenshots of your plots)**

*Actual vs. Predicted Prices (Test Set)*
![Test Set Predictions]([path/to/your/test_plot.png](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects/blob/main/Bitcoin%20Price%20Prediction%20using%20LSTM/predicted.png))

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/Amirreza-Bazmi/Your-Repo-Name.git](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects.git)
cd Deep-Learning-Projects/Bitcoin Price Prediction using LSTM
```

---

## 📬 Contact

For questions or collaborations, feel free to connect with me:

-   **📧 Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
-   **🐙 GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)
