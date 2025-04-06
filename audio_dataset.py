import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from transformers import AutoProcessor
import os
from sklearn.model_selection import train_test_split


class AudioDataset(Dataset):
    def __init__(self, data_samples, target_length=16000*3, augment=False):  # 3 seconds at 16kHz
        """
        Args:
            data_samples (list): List of tuples (audio_path, emotion_label)
            target_length (int): Target length in samples (default 3 seconds)
            augment (bool): Whether to apply audio augmentation
        """
        self.data_samples = data_samples
        self.audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.target_length = target_length
        self.augment = augment

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        audio_path, emotion_label = self.data_samples[idx]
        
        # Extract metadata from filename
        audio_file = os.path.basename(audio_path)
        parts = audio_file.split("-")
        
        # Parse all metadata
        modality = int(parts[0])
        vocal_channel = int(parts[1])
        emotion = int(parts[2]) - 1
        intensity = int(parts[3])
        statement = int(parts[4])
        repetition = int(parts[5])
        actor = int(parts[6].split(".")[0])
        gender = 0 if actor % 2 == 1 else 1

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
        
        # Process with wav2vec processor
        audio_input = self.audio_processor(
            waveform.squeeze(0), 
            sampling_rate=16000, 
            return_tensors="pt"
        )

        label = {
            "modality": modality,
            "vocal_channel": vocal_channel,
            "emotion": emotion,
            "intensity": intensity,
            "statement": statement,
            "repetition": repetition,
            "gender": gender,
        }

        return audio_input, label['emotion'], label

    def apply_augmentation(self, waveform, sample_rate):
        """
        Apply audio augmentations such as resampling, volume adjustment, and noise addition.

        Args:
            waveform (Tensor): Audio waveform
            sample_rate (int): Original sample rate

        Returns:
            Tensor: Augmented waveform
        """
        augmentations = T.Compose([
            T.Resample(orig_freq=sample_rate, new_freq=16000),
            T.Vol(0.5),  # Reduce volume
            T.AdditiveNoise(noise_factor=0.005)  # Add noise
        ])
        return augmentations(waveform)


def collate_fn(batch):
    """
    Custom collate function to handle wav2vec2 processor outputs.
    Pads the audio inputs to the longest in the batch.
    """
    try:
        audio_inputs, emotions, labels = zip(*batch)
        
        # Get max length in this batch
        max_length = max([item.input_values.shape[1] for item in audio_inputs])
        
        # Pad all inputs to max length
        padded_inputs = []
        for item in audio_inputs:
            pad_amount = max_length - item.input_values.shape[1]
            padded = torch.nn.functional.pad(
                item.input_values, 
                (0, pad_amount), 
                value=0
            )
            padded_inputs.append(padded)
        
        # Stack all padded tensors
        audio_input_values = torch.stack(padded_inputs).squeeze(1)
        
        # Stack attention masks if they exist
        if hasattr(audio_inputs[0], 'attention_mask'):
            padded_masks = []
            for item in audio_inputs:
                pad_amount = max_length - item.attention_mask.shape[1]
                padded = torch.nn.functional.pad(
                    item.attention_mask,
                    (0, pad_amount),
                    value=0
                )
                padded_masks.append(padded)
            attention_mask = torch.stack(padded_masks).squeeze(1)
        else:
            attention_mask = None
        
        # Stack other elements
        emotions = torch.tensor(emotions)
        
        return {
            'input_values': audio_input_values,
            'attention_mask': attention_mask,
            'emotions': emotions,
            'metadata': labels
        }
    except:
        return None


def load_ravdess_dataset(base_dir, test_size=0.2, random_state=42):
    """
    Load the RAVDESS dataset and split it into training and testing datasets.

    Args:
        base_dir (str): Path to the RAVDESS dataset containing actor folders.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_dataset (AudioDataset): Training dataset
        test_dataset (AudioDataset): Testing dataset
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
        collate_fn=safe_collate  # Use our custom collate function
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=safe_collate
    )
    
    return train_loader, test_loader