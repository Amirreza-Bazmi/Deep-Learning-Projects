# ✍️ Handwritten Persian Digit Recognition with CNN and Tkinter GUI

Welcome to this computer vision project! 🚀 This repository contains an end-to-end application for recognizing handwritten Persian (Farsi) digits. The project features a high-accuracy Convolutional Neural Network (CNN) trained on a Persian MNIST dataset, integrated into an interactive Graphical User Interface (GUI) built with Tkinter.

---

## 📊 Project Overview

### **Objective**
The goal of this project is to build a complete and practical application that can accurately classify handwritten Persian digits drawn by a user in real-time.

### **Key Features**
* **High-Accuracy CNN Model**: A robust Convolutional Neural Network built in Keras that achieves high performance on the digit classification task (# **98%**).
* **Interactive GUI**: A user-friendly desktop application built with **Tkinter** that allows users to upload digits and receive instant predictions.
* **Real-Time Inference**: The application loads the pre-trained model to perform predictions on new, user-generated data.

---

## 🖼️ Visual Demo


### **Model Performance**
The trained model accurately classifies digits from the test set, demonstrating strong generalization.



---

## ⚙️ Methodology

The project is divided into two main parts: model training and the GUI application.

### **1. Model Training (`training_model.ipynb`)**
* **Dataset**: The Persian MNIST dataset, containing grayscale images of digits 0-9.
* **Preprocessing**:
    * Pixel values were **resized** and **normalized** to a range of `[0, 1]`.
* **CNN Architecture**: A Sequential Keras model was designed with the following layers:
    * `Conv2D` (50 filters, 3x3 kernel, ReLU activation)
    * `MaxPooling2D` (2x2 pool size)
    * `Conv2D` (16 filters, 3x3 kernel, ReLU activation)
    * `MaxPooling2D` (2x2 pool size)
    * `Flatten`
    * `Dense` (64 units, ReLU activation)
    * `Dense` (16 units, ReLU activation)
    * `Dense` (10 units)
* **Training**: The model was trained for **10 epochs** using the `Adam` optimizer and `SparseCategoricalCrossentropy` loss function.

### **2. GUI Application (`ocr_persian.ipynb`)**
* **Framework**: The interactive user interface was built using Python's native **Tkinter** library.
* **Functionality**:
    * The model's predicted digit is displayed clearly on the UI.
* **Inference**: The application loads the pre-trained `best_accuracy_model.keras` file and uses it to predict the class of the preprocessed image drawn on the canvas.

---

## 🛠️ Technologies Used

* **Frameworks**: TensorFlow, Keras
* **GUI**: Tkinter, PIL (Pillow)
* **Libraries**: NumPy, Matplotlib, OpenCV

---

## 📈 Results

The CNN model achieved a final **test accuracy of 99.15%** after 10 epochs of training, proving its effectiveness and robustness.

![Input 1:](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects/blob/main/Handwritten%20Persian%20Digit%20Recognition/Results/input_1.png)

![Output 1:](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects/blob/main/Handwritten%20Persian%20Digit%20Recognition/Results/output_1.png)

![Input 2:](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects/blob/main/Handwritten%20Persian%20Digit%20Recognition/Results/input_2.png)

![Output 2:](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects/blob/main/Handwritten%20Persian%20Digit%20Recognition/Results/output_2.png)


---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/Amirreza-Bazmi/Your-Repo-Name.git](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects.git)
cd Deep-Learning-Projects/Handwritten Persian Digit Recognition
```

---

## 📬 Contact

For questions or collaborations, feel free to connect with me:

-   **📧 Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
-   **🐙 GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)
