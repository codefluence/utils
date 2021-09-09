import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pytorch_lightning import LightningModule


# Use autoencoder to create new features, adding along with original features to the MLP
# Train autoencoder and MLP together in each CV split to prevent data leakage
# Add target information to autoencoder (supervised learning) to force it generating more relevant features, and to create a shortcut for backpropagation of gradient
# Add Gaussian noise layer before encoder for data augmentation and to prevent overfitting


class FeatureEncoderDecoder(LightningModule):

    def __init__(self, input_width=133):

        super(FeatureEncoderDecoder, self).__init__()

        mid_width = int((input_width + 64)/2)

        self.encoder1 = nn.Linear(input_width, mid_width)
        self.encoder2 = nn.Linear(mid_width, 64)

        self.decoder1 = nn.Linear(64, mid_width)
        self.decoder2 = nn.Linear(mid_width, input_width)

        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):

        x = F.leaky_relu(self.encoder1(x))
        encoded = F.leaky_relu(self.encoder2(x))
        
        x = F.leaky_relu(self.decoder1(encoded))
        decoded = F.leaky_relu(self.decoder2(x))

        return encoded, decoded

    def training_step(self, train_batch, batch_idx):

        features, *_ = train_batch
        _, decoded = self.forward(features)
        loss = torch.mean(1 - self.cos_similarity(features, decoded))

        self.log('train_loss', loss)
        self.log('train_dist', math.degrees(np.arccos(1 - loss.detach().cpu().item())))

        return loss

    def validation_step(self, val_batch, batch_idx):

        features, *_ = val_batch
        _, decoded = self.forward(features)
        loss = torch.mean(1 - self.cos_similarity(features, decoded))

        self.log('val_dist', math.degrees(np.arccos(1 - loss.detach().cpu().item())))

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        sccheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=1/4, verbose=True, min_lr=1e-6)
        return [optimizer], {'scheduler': sccheduler, 'monitor': 'val_dist'}


