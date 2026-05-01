import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_auc_score

def denormalize(tensor):
    """Denormalize a PyTorch tensor (ImageNet stats) for visualization."""
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

def get_heatmap(anomaly_map, original_img):
    """
    Combines an anomaly map with the original image for visualization.
    anomaly_map: (H, W) numpy array
    original_img: (H, W, 3) numpy array
    """
    # Normalize anomaly map to [0, 255]
    heatmap = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay on original image
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
    return heatmap, overlay

def plot_results(img, mask, anomaly_map, overlay, save_path=None):
    """Plot original, ground truth, anomaly map, and localization."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    img_np = denormalize(img).permute(1, 2, 0).cpu().numpy()
    mask_np = mask[0].cpu().numpy() if mask.ndim == 3 else mask.cpu().numpy()
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(anomaly_map, cmap='jet')
    axes[2].set_title("Anomaly Map")
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title("Localization")
    axes[3].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_metrics(labels, scores):
    """Calculate ROC-AUC score."""
    auc = roc_auc_score(labels, scores)
    return auc
