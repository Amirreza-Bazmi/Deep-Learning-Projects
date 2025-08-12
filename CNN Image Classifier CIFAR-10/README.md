# 🖼️ CNN Image Classifier for CIFAR-10

Welcome to my project on image classification! 🚀 This repository contains a Convolutional Neural Network (CNN) built from scratch using TensorFlow and Keras to classify images from the well-known CIFAR-10 dataset. The model is also capable of predicting labels for custom, real-world images.

---

## 📊 Project Overview

### Objective
The main goal of this project is to build, train, and evaluate a robust CNN model for multi-class image classification. The project demonstrates a complete machine learning pipeline, from data loading and preprocessing to model training, evaluation, and practical inference on new images.

### Dataset
- **CIFAR-10**: A benchmark dataset consisting of 60,000 32x32 color images in 10 different classes.
  - **Classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

### Methodology
1.  **Data Loading & Preprocessing**:
    -   Loaded the CIFAR-10 dataset directly from `keras.datasets`.
    -   Normalized pixel values to a range of `[0, 1]` for better model performance.
2.  **Model Architecture**:
    -   A sequential CNN was built using Keras.
    -   **Three `Conv2D` layers** with `relu` activation for feature extraction, each followed by a `MaxPooling2D` layer to reduce dimensionality.
    -   A `Flatten` layer to convert 2D feature maps into a 1D vector.
    -   **Two `Dense` layers**, with the final layer having 10 output units for the 10 classes.
3.  **Training & Evaluation**:
    -   The model was compiled with the `adam` optimizer and `SparseCategoricalCrossentropy` loss function.
    -   Trained for 10 epochs, achieving a validation accuracy of approximately **71%**.
4.  **Inference and Real-World Testing**:
    -   The trained model was used to make predictions on the test set.
    -   More importantly, it was tested on **custom images** (e.g., a horse, a bird) to prove its practical applicability.

---

## ✨ Key Features

-   Implementation of a CNN from the ground up.
-   Complete training and validation pipeline.
-   Functionality to predict classes for any user-provided image.
-   The trained model weights are saved in `model.h5` for reusability.

---

## 🛠️ Technologies Used

-   **Framework**: TensorFlow, Keras
-   **Libraries**: NumPy, Matplotlib, OpenCV (for image processing)

---

## 📈 Results

The model was trained for 10 epochs, and the final performance on the validation set was:
-   **Validation Loss**: ~0.86
-   **Validation Accuracy**: ~71.42%

The model demonstrated its effectiveness by correctly classifying custom images not present in the original dataset.


---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/Amirreza-Bazmi/Your-Repo-Name.git](https://github.com/Amirreza-Bazmi/Deep-Learning-Projects.git)
cd Deep-Learning-Projects/CNN Image Classifier CIFAR-10
```

---

## 📬 Contact

For any questions or collaboration opportunities, feel free to reach out:

-   **📧 Email**: [bazmi.ce@gmail.com](mailto:bazmi.ce@gmail.com)
-   **🐙 GitHub**: [Amirreza-Bazmi](https://github.com/Amirreza-Bazmi)
