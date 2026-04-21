import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve

def denormalize(tensor):
    """Denormalize a PyTorch tensor to [0, 1] range."""
    device = tensor.device
    # Standard ImageNet mean/std used in transforms
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return tensor * std + mean

def get_heatmap(input_tensor, recon_tensor):
    """
    Calculate the pixel-wise difference and create a heatmap.
    Returns:
        heatmap: (H, W) numpy array
        overlay: (H, W, 3) image with heatmap overlay
    """
    # Convert to numpy and [0, 255]
    img = denormalize(input_tensor[0]).permute(1, 2, 0).cpu().detach().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    recon = denormalize(recon_tensor[0]).permute(1, 2, 0).cpu().detach().numpy()
    recon = np.clip(recon * 255, 0, 255).astype(np.uint8)
    
    # Simple MSE-like difference per pixel (averaged over channels)
    diff = np.abs(img.astype(float) - recon.astype(float)).mean(axis=2)
    
    # Normalize diff to [0, 255] for visualization
    heatmap = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    return diff, overlay

def plot_results(img, recon, mask, diff, overlay, save_path=None):
    """Plot original, reconstruction, ground truth, and heatmap."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    img_np = denormalize(img[0]).permute(1, 2, 0).cpu().detach().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    recon_np = denormalize(recon[0]).permute(1, 2, 0).cpu().detach().numpy()
    recon_np = np.clip(recon_np, 0, 1)
    
    mask_np = mask[0, 0].cpu().detach().numpy()
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(recon_np)
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')
    
    axes[2].imshow(mask_np, cmap='gray')
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    axes[3].imshow(diff, cmap='jet')
    axes[3].set_title("Diff Map")
    axes[3].axis('off')
    
    axes[4].imshow(overlay)
    axes[4].set_title("Localization")
    axes[4].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_metrics(labels, scores):
    """Calculate ROC-AUC score."""
    auc = roc_auc_score(labels, scores)
    return auc
