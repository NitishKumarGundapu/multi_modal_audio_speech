import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, AutoProcessor, CLIPVisionModel, CLIPProcessor
import pytorch_lightning as pl
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os

# Updated Dataset for RAVDESS
class RAVDESSDataset(Dataset):
    def __init__(self, audio_dir, image_dir, labels, transform=None):
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # Added CLIP processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, f"audio_{idx}.wav")
        image_path = os.path.join(self.image_dir, f"image_{idx}.jpg")
        label = self.labels[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_input = self.audio_processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt")

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = self.image_processor(images=image, return_tensors="pt").pixel_values  # Use CLIP processor

        return audio_input, image, label

# Model Class
class EmotionClassifier(pl.LightningModule):
    def __init__(self, num_classes, use_cross_attention=True):
        super(EmotionClassifier, self).__init__()
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.use_cross_attention = use_cross_attention

        # Cross-attention mechanism
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2 if not use_cross_attention else 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, audio_input, image_input):
        # Encode audio
        audio_features = self.audio_encoder(**audio_input).last_hidden_state.mean(dim=1)

        # Encode image
        image_features = self.image_encoder(pixel_values=image_input).last_hidden_state.mean(dim=1)

        # Combine features
        if self.use_cross_attention:
            audio_features = audio_features.unsqueeze(0)  # Add sequence dimension
            image_features = image_features.unsqueeze(0)
            combined_features, _ = self.cross_attention(audio_features, image_features, image_features)
            combined_features = combined_features.squeeze(0)
        else:
            combined_features = torch.cat((audio_features, image_features), dim=1)

        # Classify
        logits = self.classifier(combined_features)
        return logits

    def training_step(self, batch, batch_idx):
        audio_input, image_input, labels = batch
        logits = self(audio_input, image_input)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_input, image_input, labels = batch
        logits = self(audio_input, image_input)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# DataLoader
def create_dataloader(audio_dir, image_dir, labels, batch_size=16, transform=None):
    dataset = RAVDESSDataset(audio_dir, image_dir, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example Usage
if __name__ == "__main__":
    # Paths and labels
    audio_dir = "path_to_audio_files"
    image_dir = "path_to_image_files"
    labels = [0, 1, 2, 3, 4, 5, 6, 7]  # Example labels for emotions

    # Transform for images
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create DataLoader
    train_loader = create_dataloader(audio_dir, image_dir, labels, batch_size=16, transform=transform)

    # Initialize model
    model = EmotionClassifier(num_classes=8, use_cross_attention=True)

    # Train model
    trainer = pl.Trainer(max_epochs=10, gpus=1)
    trainer.fit(model, train_loader)