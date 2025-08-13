# 🎤 Spoken Keyword Recognition using a CNN

Welcome to my audio processing repository! 🚀 This project demonstrates how a Convolutional Neural Network (CNN) can be used to recognize and classify spoken keywords from audio clips. The model is trained to identify 8 different commands from the Mini Speech Commands dataset.

---

## 📊 Project Overview

### Objective
The goal of this project is to build an accurate audio classification model for keyword spotting. The core idea is to transform audio signals into a 2D representation (spectrograms) and then leverage the power of CNNs, typically used for image analysis, to learn and classify the distinct patterns of each spoken word.

### Dataset
-   **Mini Speech Commands Dataset**: A dataset containing thousands of one-second audio clips of 8 different keywords. The model is trained to classify the following commands:
    -   `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`

### Methodology
This project implements a sophisticated, end-to-end pipeline for audio classification:

1.  **Audio Preprocessing & Feature Extraction**:
    -   Audio files (`.wav`) are loaded and decoded into tensors.
    -   Each clip is resampled to a uniform **16kHz** using `tensorflow_io` to ensure consistency.
    -   The crucial step is converting the 1D audio signal into a 2D **spectrogram** using the Short-Time Fourier Transform (STFT). This process creates an "image" of the audio's frequency content over time, making it suitable for a CNN.
2.  **Efficient Data Pipeline (`tf.data`)**:
    -   A highly efficient data pipeline was built using `tf.data.Dataset` to handle audio files.
    -   This pipeline includes shuffling, mapping the preprocessing function, caching for performance, and batching the data for training.
3.  **Model Architecture**:
    -   A **Convolutional Neural Network (CNN)** was designed to learn features from the spectrograms.
    -   The architecture includes multiple `Conv2D` layers, each followed by `BatchNormalization` and `ReLU` activation for stable and faster training.
    -   `MaxPooling2D` layers are used for downsampling, and `Dropout` is applied for regularization.
4.  **Advanced Training Techniques**:
    -   The model was trained using the `Adam` optimizer and `SparseCategoricalCrossentropy` loss.
    -   Two key callbacks were used to optimize the training process:
        -   **`EarlyStopping`**: To prevent overfitting by stopping the training when validation performance ceases to improve.
        -   **`ReduceLROnPlateau`**: To dynamically decrease the learning rate during plateaus, helping the model to find a better minimum.
5.  **Inference Function**:
    -   A complete `predict` function is provided to take a new audio file path, apply the full preprocessing pipeline, and output the model's prediction for the spoken command.

---

## 🛠️ Technologies Used

-   **Frameworks**: TensorFlow, Keras
-   **Libraries**: TensorFlow I/O, NumPy, Matplotlib, Seaborn, OS, Pathlib

---

## 📈 Results

The model was successfully trained and evaluated on a held-out test set, demonstrating its ability to accurately classify spoken keywords. The training history plots for accuracy and loss showcase stable learning and good generalization.


*Model Output*
![Training History](path/to/your/training_plot.png)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/Amirreza-Bazmi/Your-Repo-Name.git](https://github.com/Amirreza-Bazmi/Your-Repo-Name.git)
cd Your-Repo-Name/Your-Project-Folder
```
*(Remember to update `Your-Repo-Name` and `Your-Project-Folder`)*

### 2. Install Dependencies
Ensure you have the necessary libraries installed.
```bash
pip install tensorflow tensorflow-io numpy matplotlib seaborn
```

### 3. Run the Notebook
Open and execute the `j9.ipynb` notebook. The Mini Speech Commands dataset will be downloaded automatically by the script.

---

## 📬 Contact

For any questions or collaborations, feel free to get in touch:

-   **📧 Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
-   **🐙 GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)
