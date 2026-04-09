# AI-Based Voice Emotion Detection Using Deep Learning

A machine learning system for detecting emotions from audio files using Convolutional Neural Networks (CNN) and MFCC (Mel-Frequency Cepstral Coefficients) feature extraction.

## Overview

This project implements a CNN-based classifier trained on the RAVDESS (Ryerson Audio-Visual Emotion Database and Speech) dataset to classify speech emotions into eight categories: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgusted, and Surprised.

## Features

- Audio feature extraction using MFCC coefficients
- Convolutional Neural Network with batch normalization
- Regularization through dropout layers
- Stratified train-test split for balanced evaluation
- Real-time emotion classification on audio samples
- Audio signal visualization (waveform, spectrogram, MFCC)
- Confusion matrix generation and accuracy metrics
- Model serialization in HDF5 format  

## Project Structure

```
09_AI- BASED_VOICE_EMOTION_DETECTION_USING_DL/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── Voice_Emotion_Detection.ipynb       # Main Jupyter notebook
├── voice_emotion_model.h5              # Trained model (generated)
└── RavdessData/
    ├── Actor_01/
    ├── Actor_02/
    ├── ...
    └── Actor_24/
```

**Note**: The `RavdessData/` directory is git-ignored and not included in the repository. Users must download it separately to run the notebook.

## Dataset

**RAVDESS Dataset Specifications**

- Sample Size: 7,356 audio files from 24 actors
- Emotion Classes: 8 distinct emotions
- Recordings per Emotion: 24 samples
- Audio Format: WAV
- Default Sample Rate: 16 kHz (resampled to 22,050 Hz for processing)

The RAVDESS dataset can be downloaded from: https://zenodo.org/record/1188976

## Installation

### System Requirements

- Python 3.7 or higher
- pip or conda package manager
- 4GB minimum RAM (8GB recommended for training)

### Setup Instructions

1. Navigate to project directory:
   ```bash
   cd 09_AI-BASED_VOICE_EMOTION_DETECTION_USING_DL
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   On Windows:
   ```bash
   venv\Scripts\activate
   ```

   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Prepare dataset directory:
   - The RAVDESS dataset is NOT included in this repository (git-ignored for size optimization)
   - Download the dataset from https://zenodo.org/record/1188976
   - Extract files into the `RavdessData/` folder in the project root
   - Verify folder structure: `RavdessData/Actor_01/`, `RavdessData/Actor_02/`, ..., `RavdessData/Actor_24/`
   - The directory will be automatically recognized by the notebook

## Usage

### Running the Jupyter Notebook

1. Start Jupyter from the project directory:
   ```bash
   jupyter notebook
   ```

2. Open `Voice_Emotion_Detection.ipynb` in the web browser

3. Execute cells in order:
   - Cell 1: Environment initialization
   - Cell 2: Import dependencies
   - Cell 3: Configure dataset path
   - Cells 4-6: Define feature extraction and load dataset
   - Cells 7-8: Preprocess and split data
   - Cells 9-12: Build, train, and evaluate model
   - Cells 13+: Generate predictions and visualizations

### Running in VS Code

1. Open the notebook file in VS Code
2. Select the appropriate Python interpreter (virtual environment)
3. Execute cells individually or select "Run All"

## Model Architecture

```
Input (40 MFCC features × 174 time steps × 1 channel)
↓
Conv2D (64 filters, 3×3) + BatchNorm + ReLU
↓
MaxPooling2D (2×2)
↓
Dropout (0.3)
↓
Conv2D (128 filters, 3×3) + BatchNorm + ReLU
↓
MaxPooling2D (2×2)
↓
Dropout (0.3)
↓
Flatten
↓
Dense (128, ReLU) + Dropout (0.3)
↓
Dense (8, Softmax) → Output
```

## Model Parameters

- **CNN Layers**: 2 convolutional blocks
- **Feature Extraction**: 40 MFCC coefficients
- **Max Padding Length**: 174 time steps
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Epochs**: 30
- **Batch Size**: 32
- **Validation Split**: 20% (stratified)

## Results

The model outputs:
- **Confusion Matrix**: Shows prediction accuracy per emotion class
- **Model Accuracy**: Overall accuracy on test set
- **Training History**: Plots of accuracy and loss over epochs
- **Sample Predictions**: Real-time emotion detection on test audio files

## Key Functions

### `extract_features(file_path, max_pad_len=174)`
- Loads audio file using librosa
- Extracts 40 MFCC coefficients
- Pads or truncates to max_pad_len
- Returns shape (40, 174)

### `visualize_audio(audio_path)`
- Displays waveform
- Shows spectrogram
- Plots MFCC heatmap

### `plot_confusion_matrix(y_true, y_pred, labels)`
- Creates heatmap of model predictions vs actual labels
- Helps identify misclassified emotions

## Modifications for Local Execution

The original notebook was designed for Google Colab. For **local execution** in VS Code:

- Google Colab mount is optional (wrapped in try-except)
- Dataset path automatically uses local `RavdessData/` folder
- Works on Windows, macOS, and Linux

## Performance Optimization

GPU Acceleration

To significantly improve training speed (approximately 10x faster), install the GPU-enabled version of TensorFlow:

```bash
pip install tensorflow-gpu
```

Consider the following optimizations:

- Start with a subset of the dataset (e.g., 5 actors) for rapid experimentation
- Increase the number of epochs beyond 30 for improved accuracy
- Reduce batch size if memory constraints are encountered
- Adjust dropout rates (currently 0.3) for fine-tuned regularization

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing modules | Execute `pip install -r requirements.txt` to install all dependencies |
| Dataset not found | Verify `RavdessData/` folder exists with uncompressed `.wav` files |
| Memory errors during training | Reduce batch_size parameter or train on fewer actor folders |
| Slow training without GPU | Install tensorflow-gpu for hardware acceleration |
| Model loading errors | Ensure `voice_emotion_model.h5` exists in the working directory |

## Future Enhancements

Potential improvements and extensions:

- Evaluate alternative CNN architectures (ResNet, VGG, Inception)
- Implement data augmentation techniques (time-shifting, pitch-shifting, noise injection)
- Develop real-time emotion detection from microphone input
- Deploy model as a REST API using Flask or FastAPI
- Cross-dataset validation with TESS and CREMA-D datasets
- Implement transfer learning using pre-trained models
- Add attention mechanisms for improved interpretability

## References

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Emotion Database (RAVDESS). PLOS ONE, 13(5), e0196424.
- McFee, B., Raffel, C., Liang, D., et al. (2015). Librosa: Audio and music signal analysis in Python. SciPy Conference Proceedings.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.