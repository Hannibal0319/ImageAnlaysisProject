import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dataset import get_dataloader
from model import get_model
from utils import denormalize, get_heatmap, plot_results

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloader
    dataloader = get_dataloader(args.root_dir, args.category, split='test', batch_size=1, shuffle=False, resolution=args.resolution)
    
    # Load Frozen Feature Extractor
    model = get_model().to(device)
    model.eval()
    
    # Load Memory Bank
    bank_path = os.path.join(args.checkpoint_dir, f"{args.category}_memory_bank.pth")
    if not os.path.exists(bank_path):
        print(f"Memory bank not found at {bank_path}. Please build it first using train.py.")
        return
        
    print(f"Loading memory bank for {args.category}...")
    memory_bank = torch.load(bank_path, map_location='cpu').numpy()
    
    # Initialize Nearest Neighbors
    print("Fitting Nearest Neighbors index...")
    knn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
    knn.fit(memory_bank)
    
    image_scores = []
    image_labels = []
    all_pixel_scores = []
    all_pixel_labels = []
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    print(f"Evaluating {args.category} (Frozen PatchCore)...")
    with torch.no_grad():
        for i, (images, label, masks, names) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            
            # Extract features (multi-scale)
            features = model(images)
            
            # Aggregate spatial context (3x3 average pooling to make features locally aware)
            for j, f in enumerate(features):
                f_pooled = F.avg_pool2d(f, 3, stride=1, padding=1)
                features[j] = f_pooled
            
            f1, f2 = features
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            combined = torch.cat([f1, f2_up], dim=1) # [1, C, H_feat, W_feat]
            
            # Reshape for NN search
            h_feat, w_feat = combined.shape[-2:]
            test_features = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1]).cpu().numpy()
            
            # Find distances to the nearest normal neighbor
            distances, _ = knn.kneighbors(test_features) # [H_feat*W_feat, 1]
            
            # Reshape to 2D anomaly map
            anomaly_map = distances.reshape(h_feat, w_feat)
            
            # Upsample anomaly map to original image resolution
            anomaly_map_resized = cv2.resize(anomaly_map, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
            
            # Gaussian blur to smooth
            anomaly_map_resized = cv2.GaussianBlur(anomaly_map_resized, (3, 3), 0)
            
            # Image score
            image_score = np.max(anomaly_map_resized)
            image_scores.append(image_score)
            image_labels.append(label.item())
            
            # Pixel-level labels
            mask_np = masks[0, 0].cpu().numpy()
            mask_np = (mask_np > 0.5).astype(int)
            all_pixel_scores.append(anomaly_map_resized.flatten())
            all_pixel_labels.append(mask_np.flatten())
            
            # Visualization
            if i < 20: 
                img_np = denormalize(images[0]).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                heatmap, overlay = get_heatmap(anomaly_map_resized, img_np)
                save_path = os.path.join(args.result_dir, f"{args.category}_{i}_{names[0]}")
                plot_results(images, masks, anomaly_map_resized, overlay, save_path=save_path)
                
    # --- Metrics ---
    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)
    img_auc = roc_auc_score(image_labels, image_scores)
    
    all_pixel_scores = np.concatenate(all_pixel_scores)
    all_pixel_labels = np.concatenate(all_pixel_labels)
    pixel_auc = roc_auc_score(all_pixel_labels, all_pixel_scores)
    
    print(f"\n{args.category} Final Metrics (Frozen PatchCore):")
    print(f"  > Image-level AUROC: {img_auc:.4f}")
    print(f"  > Pixel-level AUROC: {pixel_auc:.4f} (Localization)")
    
    with open(os.path.join(args.result_dir, "metrics.txt"), "a") as f:
        f.write(f"{args.category} (Frozen): Image_AUC={img_auc:.4f}, Pixel_AUC={pixel_auc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Frozen Feature Matching for Anomaly Detection")
    parser.add_argument("--root_dir", type=str, default="mvtec_ad", help="Path to MVTec AD dataset")
    parser.add_argument("--category", type=str, default="bottle", help="Category to evaluate")
    parser.add_argument("--resolution", type=int, default=224, help="Image resolution")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory where memory bank is saved")
    parser.add_argument("--result_dir", type=str, default="results", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    evaluate(args)
