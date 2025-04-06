
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import HubertModel
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from peft import get_peft_model, LoraConfig, TaskType  # Import PEFT components
from audio_dataset import get_data_loaders


class AudioEmotionClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(AudioEmotionClassifier, self).__init__()
        # Load the Wav2Vec2 model

        
        # self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        # for param in self.audio_encoder.parameters():
        #     param.requires_grad = True

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, audio_input):
        # Extract audio features using the PEFT-enabled Wav2Vec2 model
        audio_features = self.audio_encoder(audio_input).last_hidden_state.mean(dim=1)
        logits = self.classifier(audio_features)
        return logits

    def training_step(self, batch, batch_idx):
        audio_input, labels = batch['input_values'], batch['emotions']
        logits = self(audio_input)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_input, labels = batch['input_values'], batch['emotions']
        logits = self(audio_input)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


if __name__ == "__main__":
    # Path to the RAVDESS dataset
    audio_dir = r"C:\Users\gnith\Desktop\multi_modal_audio_speech\ravdess_dataset"

    # Get train and validation dataloaders
    train_loader, val_loader = get_data_loaders(audio_dir, batch_size=2)

    # Initialize the model
    model = AudioEmotionClassifier(num_classes=8)

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=5, devices=1)

    # Train the model
    trainer.fit(model, train_loader, val_loader)