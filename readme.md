# Multi-Modal Audio Speech Emotion Classification

This project implements a multi-modal audio speech emotion classification system using PyTorch Lightning and Vision Transformers (ViT). The system processes audio data, converts it into spectrograms, and uses a pre-trained Vision Transformer model to classify emotions.

## Features

1. **Audio Preprocessing**:
   - Converts audio files into spectrograms using Mel-Spectrogram transformation.
   - Normalizes and resizes spectrograms to fit the input requirements of the Vision Transformer.

2. **Model Architecture**:
   - Utilizes a pre-trained Vision Transformer (ViT) from the `timm` library.
   - Adds a custom classification head for emotion classification.
   - Supports fine-tuning of the ViT model with a lower learning rate for pre-trained layers.

3. **Dataset Handling**:
   - Processes the RAVDESS dataset for emotion classification.
   - Splits the dataset into training and validation sets.
   - Ensures labels are zero-indexed and compatible with `nn.CrossEntropyLoss`.

4. **Training and Validation**:
   - Implements training and validation steps using PyTorch Lightning.
   - Logs training and validation losses and accuracy metrics.
   - Supports GPU acceleration for faster training.

5. **Optimization**:
   - Uses the AdamW optimizer with separate learning rates for the ViT and classification head.
   - Includes a cosine annealing learning rate scheduler.

## File Structure

- **`specto_main_pl.py`**:
  - Contains the main model definition (`AudioEmotionClassifier`) and training script.
  - Defines the Vision Transformer-based architecture and training pipeline.

- **`spectogram_dataset.py`**:
  - Handles dataset loading and preprocessing.
  - Converts audio files into spectrograms and prepares them for training.
  - Includes data augmentation options.

- **`image_dataset.py`**:
  - Processes video data into frames and saves them as pickle files for further use.

- **`test.ipynb`**:
  - Contains test scripts and experiments for debugging and validating the dataset and model.

## How It Works

1. **Dataset Preparation**:
   - The RAVDESS dataset is loaded and split into training and validation sets.
   - Audio files are converted into spectrograms using the `AudioDataset` class in `spectogram_dataset.py`.

2. **Model Training**:
   - The `AudioEmotionClassifier` model is initialized with a pre-trained ViT backbone.
   - The model is trained using PyTorch Lightning's `Trainer` class.
   - Training and validation metrics are logged for monitoring.

3. **Inference**:
   - The trained model can classify emotions from spectrograms generated from audio files.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- `timm` library for Vision Transformers
- `torchaudio` for audio processing
- `scikit-learn` for dataset splitting

## Usage

1. **Prepare the Dataset**:
   - Place the RAVDESS dataset in the `ravdess_dataset_1` folder.

2. **Run the Training Script**:
   ```bash
   python specto_main_pl.py