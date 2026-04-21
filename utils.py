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

def generate_synthetic_anomaly(images, p=0.5):
    """
    Randomly adds synthetic anomalies to a batch of images.
    Returns:
        corrupted_images: Tensor of the same shape as images
        labels: Tensor (batch,) with 1 if anomaly was added, else 0
    """
    corrupted_images = images.clone()
    batch_size, channels, h, w = images.shape
    labels = torch.zeros(batch_size, device=images.device)
    
    for i in range(batch_size):
        if torch.rand(1).item() < p:
            # Add anomaly
            labels[i] = 1.0
            # Random patch size between 5% and 15% of image size
            ph = torch.randint(int(h * 0.05), int(h * 0.15), (1,)).item()
            pw = torch.randint(int(w * 0.05), int(w * 0.15), (1,)).item()
            
            # Random position
            py = torch.randint(0, h - ph, (1,)).item()
            px = torch.randint(0, w - pw, (1,)).item()
            
            # Randomly choose anomaly type: random noise, intensity shift, or zeroing out
            anomaly_type = torch.randint(0, 3, (1,)).item()
            if anomaly_type == 0:
                # Random noise
                corrupted_images[i, :, py:py+ph, px:px+pw] = torch.rand(channels, ph, pw, device=images.device)
            elif anomaly_type == 1:
                # Intensity shift
                corrupted_images[i, :, py:py+ph, px:px+pw] += 0.5 * torch.randn(1, device=images.device)
            else:
                # Zeroing out (simulating a hole/obstruction)
                corrupted_images[i, :, py:py+ph, px:px+pw] = 0
                
    return corrupted_images, labels
