import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import HubertModel
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from peft import get_peft_model, LoraConfig, TaskType  # Import PEFT components
from multimodal_dataset import *


class ImageEmotionClassifier(pl.LightningModule):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=True, class_weights=None):
        super(ImageEmotionClassifier, self).__init__()
        
        # Save class weights as a buffer
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None)
        
        # Load ViT model from timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # We'll add our own head
            in_chans=12  # Video data has 0 input channels
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

    def _init_weights(self, module):
        """Initialize weights for linear layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Normalize input
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
        
        # Get features from ViT
        features = self.vit(x)
        
        # Classify
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        try:
            video = batch['video']
            labels = batch['emotion']
            logits = self(video)
            
            # Use class weights in CrossEntropyLoss
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_epoch=True, prog_bar=True)
            return loss
        except:
            return 0

    def validation_step(self, batch, batch_idx):
        try:
            video = batch['video']
            labels = batch['emotion']
            logits = self(video)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
            return loss
        except:
            return 0

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


multimodel_dataloader = MultimodalDataProcessor("multimodal_dataset_temp.pickle", batch_size=2)
train_dataloader, val_dataloader, test_dataloader, class_weights = multimodel_dataloader.process()

model = ImageEmotionClassifier(num_classes=8,class_weights=class_weights.numpy())
trainer = pl.Trainer(max_epochs=5, devices=1, accelerator='gpu')
trainer.fit(model, train_dataloader, val_dataloader)