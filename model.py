import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Use pre-trained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        
        self.captured_features = []
        # Register hooks to extract intermediate Layer 2 and Layer 3 features
        self.model.layer2.register_forward_hook(self.hook_fn)
        self.model.layer3.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.captured_features.append(output)

    def forward(self, x):
        self.captured_features = []
        _ = self.model(x)
        return self.captured_features

def get_model():
    """Returns the frozen, pre-trained feature extractor."""
    return FeatureExtractor()
