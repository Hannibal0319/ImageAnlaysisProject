import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, dropout_p=0.2):
        super(CNNAutoencoder, self).__init__()
        self.dropout_p = dropout_p
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (3, 256, 256)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
        )
        
        # Decode bottleneck to reconstruct
        self.decoder = nn.Sequential(
            # Input: (512, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout_p),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # (3, 256, 256)
            nn.Tanh() # Use Tanh to squash output if using Normalized inputs in range [~-1, 1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def get_model(latent_dim=128, dropout_p=0.2):
    return CNNAutoencoder(latent_dim=latent_dim, dropout_p=dropout_p)
