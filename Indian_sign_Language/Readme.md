# Indian Sign Language (ISL) Recognition

## Overview

This project aims to recognize Indian Sign Language (ISL) gestures using computer vision and machine learning techniques. The model is trained on a dataset of ISL gestures and uses an SVM classifier for real-time hand gesture recognition.

## Features

- Extracts hand landmarks using MediaPipe
- Supports two-hand recognition
- Trains an SVM model with a standardized dataset
- Performs real-time ISL gesture recognition via webcam
- Provides visual feedback with detected gestures and confidence scores

## Dataset

**Dataset Download Link:** https\://www\.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl?resource=download\
The dataset contains labeled images of Indian Sign Language gestures.

## Installation

### Prerequisites

Ensure you have Python installed along with the following dependencies:

```bash
pip install opencv-python numpy mediapipe scikit-learn
```

## Usage

### 1. Extract Dataset

Ensure the dataset ZIP file is placed in the project directory. The script will automatically extract it.

### 2. Train the Model

Run the following command to train the ISL model:

```bash
python IndianSignLanguage.py
```

This will extract the dataset, train an SVM model, and save it as `isl_svm_model_two_hands.pkl`.

### 3. Real-Time Gesture Recognition

After training, the program will automatically start real-time gesture recognition. Press 'q' to exit.

## File Structure

```
📂 Indian-Sign-LanguageISL
 ├── 📂 Indian  # Extracted dataset
 ├── IndianSignLanguage.py  # Main script
 ├── isl_svm_model_two_hands.pkl  # Trained model
 ├── isl_scaler_two_hands.pkl  # StandardScaler for features
```

## Contributing

Feel free to open an issue or submit a pull request to enhance the project.


