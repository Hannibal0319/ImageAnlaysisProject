import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomAutoencoder(nn.Module):
    def __init__(self):
        super(CustomAutoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            #nn.Sigmoid() this might need to be activated depending on the input normalization
        )

    def encode(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        return f2, f3

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)

        x = self.dec3(f3)
        x = self.dec2(x)
        x = self.dec1(x)

        return x


class CustomCNNFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(CustomCNNFeatureExtractor, self).__init__()

        autoencoder = CustomAutoencoder()

        if checkpoint_path is not None:
            autoencoder.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        self.enc1 = autoencoder.enc1
        self.enc2 = autoencoder.enc2
        self.enc3 = autoencoder.enc3

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)

        return [f2, f3]


def get_model():
    return CustomCNNFeatureExtractor(
        checkpoint_path="checkpoints/custom_autoencoder.pth"
    )