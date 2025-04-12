import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import timm 
from multimodal_dataset import MultimodalDataProcessor  # Assuming you have a dataset class for loading your data


class AudioEmotionClassifier(pl.LightningModule):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=True, class_weights=None):
        super(AudioEmotionClassifier, self).__init__()
        
        # Load ViT model from timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # We'll add our own head
            in_chans=1  # Our spectrograms have 1 channel
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
        
        # Class weights for weighted loss
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

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
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        # Extract audio and emotion from the multimodal dataset
        audio = batch['audio']
        labels = batch['emotion']
        
        # Forward pass
        logits = self(audio)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device) if self.class_weights is not None else None)
        loss = loss_fn(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Extract audio and emotion from the multimodal dataset
        audio = batch['audio']
        labels = batch['emotion']
        
        # Forward pass
        logits = self(audio)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device) if self.class_weights is not None else None)
        loss = loss_fn(logits, labels)
        
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
    # Path to the multimodal dataset
    multimodel_dataloader = MultimodalDataProcessor("multimodal_dataset_temp.pickle", batch_size=2)
    train_dataloader, val_dataloader, test_dataloader, class_weights = multimodel_dataloader.process() 

    # Initialize the model
    model = AudioEmotionClassifier(num_classes=8, model_name='vit_base_patch16_224', pretrained=True, class_weights=class_weights)

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=10,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # precision=16  # Mixed precision training
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)