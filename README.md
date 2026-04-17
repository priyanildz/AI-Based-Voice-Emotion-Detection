<div align="center">

# <img src="https://img.icons8.com/fluency/48/artificial-intelligence.png" width="40"/> AI-Based Voice Emotion Detection

### Deep Learning • Audio Analysis • Emotion Classification

<p>
A deep learning-based application that detects human emotions from voice input by analyzing audio features and classifying them into distinct emotional categories.
</p>

<br/>

<a href="YOUR_LIVE_LINK_HERE" target="_blank">
  <img src="https://img.shields.io/badge/Live%20Application-Open-1E88E5?style=for-the-badge&logo=google-chrome&logoColor=white" />
</a>

<br/><br/>

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Deep%20Learning-Model-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Librosa-Audio%20Processing-000000?style=for-the-badge"/>
<img src="https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-2E7D32?style=for-the-badge"/>

</div>

---

## Overview

**AI-Based Voice Emotion Detection** is a deep learning project that analyzes audio signals to identify human emotions such as happiness, sadness, anger, fear, and more.

The system extracts features from voice recordings and uses a trained model to classify emotions, making it useful for applications in human-computer interaction, mental health analysis, and voice-based systems.

---

## Screenshots & Model Analysis

<div align="center">

| Confusion Matrix | 
|------------------|
| <img src="assets/confusion_matrix.png" width="800"/> |
| Accuracy & Loss Graph |
| <img src="assets/output.png" width="800"/> |

</div>

---

## Explanation of Visualizations

### 1. Confusion Matrix

The confusion matrix shows how well the model classifies different emotions.

- Rows represent **actual emotions**  
- Columns represent **predicted emotions**  
- Diagonal values indicate **correct predictions**  
- Off-diagonal values indicate **misclassifications**  

From the matrix:
- Some emotions like **Happy** and **Neutral** are predicted correctly  
- Certain emotions (like Fearful, Angry) may overlap, showing model confusion  
- Helps evaluate model performance in detail beyond accuracy  

---

### 2. Model Accuracy Graph

This graph shows how accuracy improves over training epochs.

- Training accuracy steadily increases  
- Validation accuracy follows a similar trend  
- Indicates the model is learning patterns from data  

A close alignment between training and validation accuracy suggests **no severe overfitting**.

---

### 3. Model Loss Graph

This graph shows how error decreases over time.

- Training loss decreases rapidly  
- Validation loss also decreases gradually  
- Stabilization indicates the model is converging  

Lower loss means better prediction performance.

---

## Key Features

- Emotion detection from voice/audio input  
- Deep learning-based classification  
- Feature extraction using audio processing  
- Model training and evaluation  
- Visualization of performance metrics  
- Multi-class emotion classification  

---

## Technology Stack

<div align="center">

| Category | Technology |
|----------|-----------|
| Language | <img src="https://img.icons8.com/color/20/python.png"/> Python |
| Deep Learning | <img src="https://img.icons8.com/color/20/artificial-intelligence.png"/> TensorFlow / Keras |
| Audio Processing | <img src="https://img.icons8.com/ios-filled/20/sound.png"/> Librosa |
| Numerical | <img src="https://img.icons8.com/color/20/numpy.png"/> NumPy |
| Visualization | <img src="https://img.icons8.com/color/20/combo-chart.png"/> Matplotlib |

</div>

---

## Project Structure

```
09_voice_emotion_detection/
├── dataset/
│   └── audio_files/
├── model/
│   └── trained_model.h5
├── notebooks/
│   └── training.ipynb
├── utils/
│   └── feature_extraction.py
├── requirements.txt
└── assets/
    ├── confusion_matrix.png
    └── training_metrics.png
```

---

## How It Works

1. Audio input is collected  
2. Features are extracted (e.g., MFCCs using Librosa)  
3. Data is passed into deep learning model  
4. Model predicts emotion category  
5. Output is classified into predefined labels  

---

## Model Details

- Multi-class classification model  
- Trained on labeled emotional audio dataset  
- Learns patterns in frequency, tone, and pitch  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Jupyter Notebook  

---

### Installation

```bash
git clone https://github.com/priyanildz/Voice-Emotion-Detection.git
cd Voice-Emotion-Detection
```

```bash
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
jupyter notebook
```

Run the training or prediction notebook.

---

## Use Cases

- Emotion-aware virtual assistants  
- Call center sentiment analysis  
- Mental health monitoring  
- Human-computer interaction systems  

---

## Future Improvements

- Real-time voice input detection  
- Web or mobile interface  
- Improved model accuracy with larger dataset  
- Deployment using Flask or Streamlit  

---

## License

This project is licensed under the MIT License.

---

<div align="center">

Developed by  
<strong>priyanildz</strong>

</div>