import torch
import torch.nn as nn
try:
    import torchvision.models as models
    USE_TIMM = False
except ImportError:
    import timm
    USE_TIMM = True

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

        self.enc3 = nn.Sequential( #test
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

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        global USE_TIMM
        if not USE_TIMM:
            try:
                # Use pre-trained ResNet18
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                import timm
                self.model = timm.create_model('resnet18', pretrained=True)
                USE_TIMM = True
        else:
            self.model = timm.create_model('resnet18', pretrained=True)
            
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        
        self.captured_features = []
        # Register hooks to extract intermediate Layer 2 and Layer 3 features
        if not USE_TIMM:
            self.model.layer2.register_forward_hook(self.hook_fn)
            self.model.layer3.register_forward_hook(self.hook_fn)
        else:
            # Timm resnet structure
            self.model.layer2.register_forward_hook(self.hook_fn)
            self.model.layer3.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.captured_features.append(output)


def get_model(which="resnet18"):
    if which == "custom_autoencoder" or which == "autoencoder":
        return CustomCNNFeatureExtractor(
            checkpoint_path="checkpoints/custom_autoencoder.pth"
        )
    elif which == "resnet18" or which == "can_memory_bank":
        return FeatureExtractor()
    else:
        raise ValueError("Invalid model type. Choose 'custom_autoencoder' or 'resnet18'.")
