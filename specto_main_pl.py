import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import timm  # Import timm library for ViT
from spectogram_dataset import get_data_loaders  # Make sure this imports the spectrogram version


class AudioEmotionClassifier(pl.LightningModule):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=True):
        super(AudioEmotionClassifier, self).__init__()
        
        # Load ViT model from timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # We'll add our own head
            in_chans=3  # Our spectrograms have 3 channels
        )
        
        # Get the feature dimension of the ViT
        self.feature_dim = self.vit.num_features
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights for classifier
        self._init_weights(self.classifier)
        
        # Spectrogram stats for normalization (adjust based on your data)
        self.register_buffer('spectrogram_mean', torch.tensor([0.5]))
        self.register_buffer('spectrogram_std', torch.tensor([0.5]))

    def _init_weights(self, module):
        """Initialize weights for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Normalize spectrograms (assuming they're in [0,1] range)
        x = (x - self.spectrogram_mean) / self.spectrogram_std
        
        # Get features from ViT
        features = self.vit(x)
        
        # Classify
        # logits = self.classifier(features)
        return features

    def training_step(self, batch, batch_idx):
        spectrograms, labels = batch['spectrograms'], batch['emotions']
        logits = self(spectrograms)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        # self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        spectrograms, labels = batch['spectrograms'], batch['emotions']
        logits = self(spectrograms)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Separate parameters for the ViT and classifier
        vit_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'vit' in name:
                if 'mlp' in name:
                    vit_params.append(param)
                else:
                    param.requires_grad = False
            else:
                classifier_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': vit_params, 'lr': 1e-5},  # Lower learning rate for pretrained ViT
            {'params': classifier_params, 'lr': 1e-4}  # Higher learning rate for new head
        ], weight_decay=0.01)
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        return [optimizer], [scheduler]


if __name__ == "__main__":
    # Path to the RAVDESS dataset
    audio_dir = r"C:\Users\gnith\Desktop\multi_modal_audio_speech\ravdess_dataset_1"

    # Get train and validation dataloaders (make sure this uses the spectrogram version)
    train_loader, val_loader = get_data_loaders(audio_dir, batch_size=2)  # Can use larger batch size with ViT

    # Initialize the model
    model = AudioEmotionClassifier(num_classes=8, model_name='vit_base_patch16_224', pretrained=True)

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=20,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # precision=16  # Mixed precision training
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)