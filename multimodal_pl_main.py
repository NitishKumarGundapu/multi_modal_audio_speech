import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
from multimodal_dataset import MultimodalDataProcessor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('medium')

class MultimodalEmotionClassifier(pl.LightningModule):
    def __init__(self, num_classes, audio_model_name='vit_base_patch16_224', video_model_name='vit_base_patch16_224', pretrained=True, class_weights=None):
        super(MultimodalEmotionClassifier, self).__init__()

        # Save class weights as a buffer
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None)

        # Audio model (ViT for spectrograms)
        self.audio_vit = timm.create_model(
            audio_model_name,
            pretrained=pretrained,
            num_classes=0,  # We'll add our own head
            in_chans=3  # Spectrograms have 3 channels
        )
        self.audio_feature_dim = self.audio_vit.num_features

        # Video model (ViT for video frames)
        self.video_vit = timm.create_model(
            video_model_name,
            pretrained=pretrained,
            num_classes=0,  # We'll add our own head
            in_chans=12  # Video frames have 10 channels (RGB)
        )
        self.video_feature_dim = self.video_vit.num_features

        # Combined feature dimension
        self.combined_feature_dim = self.audio_feature_dim + self.video_feature_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.combined_feature_dim),
            nn.Linear(self.combined_feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Initialize weights for the classifier
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize weights for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, audio, video):
        # Process audio input
        audio_features = self.audio_vit(audio)

        # Process video input
        video_features = self.video_vit(video)

        # Concatenate audio and video features
        combined_features = torch.cat((audio_features, video_features), dim=1)

        # Classification
        logits = self.classifier(combined_features)
        return logits

    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        video = batch['video']
        labels = batch['emotion']

        # Forward pass
        logits = self(audio, video)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device) if self.class_weights is not None else None)
        loss = loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch['audio']
        video = batch['video']
        labels = batch['emotion']

        # Forward pass
        logits = self(audio, video)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device) if self.class_weights is not None else None)
        loss = loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Separate parameters for the ViT models and classifier
        audio_vit_params = []
        video_vit_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            if 'audio_vit' in name:
                audio_vit_params.append(param)
            elif 'video_vit' in name:
                video_vit_params.append(param)
            else:
                classifier_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': audio_vit_params, 'lr': 1e-5},  # Lower learning rate for pretrained audio ViT
            {'params': video_vit_params, 'lr': 1e-5},  # Lower learning rate for pretrained video ViT
            {'params': classifier_params, 'lr': 1e-5}  # Higher learning rate for new head
        ], weight_decay=0.01)

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    # Path to the multimodal dataset
    multimodel_dataloader = MultimodalDataProcessor("multimodal_dataset_temp.pickle", batch_size=2)
    train_dataloader, val_dataloader, test_dataloader, class_weights = multimodel_dataloader.process()

    # Initialize the model
    model = MultimodalEmotionClassifier(
        num_classes=8,
        audio_model_name='vit_base_patch16_224',
        video_model_name='vit_base_patch16_224',
        pretrained=True,
        class_weights=class_weights
    )

    # Early stopping callback
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=10,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[early_stopping],
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)