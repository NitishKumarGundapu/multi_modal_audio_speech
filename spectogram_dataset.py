import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from transformers import AutoProcessor
import os
from sklearn.model_selection import train_test_split
import numpy as np


class AudioDataset(Dataset):
    def __init__(self, data_samples, target_length=16000*3, augment=False, n_mels=128, n_fft=1024, hop_length=512):
        """
        Args:
            data_samples (list): List of tuples (audio_path, emotion_label)
            target_length (int): Target length in samples (default 3 seconds)
            augment (bool): Whether to apply audio augmentation
            n_mels (int): Number of Mel bins
            n_fft (int): FFT window size
            hop_length (int): Hop length between STFT windows
        """
        self.data_samples = data_samples
        self.target_length = target_length
        self.augment = augment
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Spectrogram transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0  # Use power spectrum (magnitude squared)
        )
        
        # For log compression
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        audio_path, emotion_label = self.data_samples[idx]
        
        # Ensure emotion labels are zero-indexed
        emotion_label = emotion_label - 1  # Assuming labels start from 1

        # Load and process audio
        waveform, sample_rate = torchaudio.load(audio_path, format="wav")
        
        # Apply augmentation
        if self.augment:
            waveform = self.apply_augmentation(waveform, sample_rate)

        # Normalize the waveform (zero mean, unit variance)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Pad or truncate to target length
        if waveform.shape[1] < self.target_length:
            # Pad with zeros
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > self.target_length:
            # Random crop
            start = torch.randint(0, waveform.shape[1] - self.target_length, (1,)).item()
            waveform = waveform[:, start:start+self.target_length]
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Generate Mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Apply log compression
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Normalize spectrogram to [0, 1] range
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        
        # Resize spectrogram to 224x224
        resized_spec = torch.nn.functional.interpolate(
            log_mel_spec.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Convert to 3-channel "image" by repeating the spectrogram
        # (ViT typically expects 3-channel input)
        spectrogram_img = resized_spec.repeat(3, 1, 1)
        
        return spectrogram_img, emotion_label

    def apply_augmentation(self, waveform, sample_rate):
        """
        Apply audio augmentations suitable for spectrogram generation.
        """
        # Resample to 16kHz
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Apply time masking
        waveform = torchaudio.transforms.TimeMasking(time_mask_param=100)(waveform)

        # Apply frequency masking
        waveform = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)(waveform)

        # Adjust volume
        waveform = waveform * 0.5  # Equivalent to T.Vol(0.5)

        return waveform


def collate_fn(batch):
    spectrograms, emotions = zip(*batch)
    
    # Stack spectrograms
    spectrograms = torch.stack(spectrograms)
    
    # Stack emotions
    emotions = torch.tensor(emotions)
    
    return {
        'spectrograms': spectrograms,
        'emotions': emotions
    }


def load_ravdess_dataset(base_dir, test_size=0.2, random_state=42):
    """
    Load the RAVDESS dataset and split it into training and testing datasets.
    """
    data = []
    for actor_folder in os.listdir(base_dir):
        actor_path = os.path.join(base_dir, actor_folder)
        if os.path.isdir(actor_path):
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith(".wav"):
                    parts = audio_file.split("-")
                    emotion = int(parts[2])
                    audio_path = os.path.join(actor_path, audio_file)
                    data.append((audio_path, emotion))

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    train_dataset = AudioDataset(train_data)
    test_dataset = AudioDataset(test_data)

    return train_dataset, test_dataset


def get_data_loaders(base_dir, batch_size=8, test_size=0.2, random_state=42, augment=False):
    train_dataset, test_dataset = load_ravdess_dataset(base_dir, test_size, random_state)

    train_dataset.augment = augment

    def safe_collate(batch):
        batch = [item for item in batch if item is not None]
        return collate_fn(batch)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=safe_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=safe_collate
    )
    
    return train_loader, test_loader