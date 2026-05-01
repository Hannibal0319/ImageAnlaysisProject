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
from scipy.ndimage import label

def calculate_aupro(all_pixel_scores, all_pixel_labels, fpr_limit=0.3):
    """
    Calculate Area Under Per-Region Overlap (AU-PRO).
    all_pixel_scores: List of 1D arrays of anomaly scores per pixel.
    all_pixel_labels: List of 1D arrays of binary labels per pixel.
    """
    res = int(np.sqrt(all_pixel_scores[0].shape[0]))
    scores_maps = [s.reshape(res, res) for s in all_pixel_scores]
    labels_maps = [l.reshape(res, res) for l in all_pixel_labels]
    
    all_regions = []
    for l_map in labels_maps:
        labeled, n_regions = label(l_map)
        all_regions.append((labeled, n_regions))
    
    flat_scores = np.concatenate(all_pixel_scores)
    # Use fewer thresholds for speed
    thresholds = np.percentile(flat_scores, np.linspace(0, 100, 100))
    thresholds = np.flip(thresholds)
    
    pro_curve = []
    fpr_curve = []
    total_normal_pixels = np.sum([np.sum(l == 0) for l in labels_maps])
    
    for th in thresholds:
        region_recalls = []
        fp_pixels = 0
        for i, (labeled, n_regions) in enumerate(all_regions):
            pred_mask = (scores_maps[i] >= th)
            for r in range(1, n_regions + 1):
                region_mask = (labeled == r)
                overlap = np.sum(pred_mask[region_mask])
                region_recalls.append(overlap / np.sum(region_mask))
            fp_pixels += np.sum(pred_mask[labels_maps[i] == 0])
            
        pro_curve.append(np.mean(region_recalls) if region_recalls else 1.0)
        fpr_curve.append(fp_pixels / total_normal_pixels)
        if fpr_curve[-1] > fpr_limit: break
            
    # Integrate PRO curve using trapezoidal rule
    pro_curve = np.array(pro_curve)
    fpr_curve = np.array(fpr_curve)
    fpr_curve = fpr_curve / (fpr_curve[-1] + 1e-10) # Normalize to max reached FPR
    
    # Manual trapezoidal integration
    aupro = 0
    for i in range(len(fpr_curve) - 1):
        aupro += (pro_curve[i] + pro_curve[i+1]) / 2 * (fpr_curve[i+1] - fpr_curve[i])
    return aupro

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
    
    # Initialize Nearest Neighbors with K=9 for robust weighting
    print("Fitting Nearest Neighbors index...")
    knn = NearestNeighbors(n_neighbors=9, algorithm='auto', n_jobs=-1)
    knn.fit(memory_bank)
    
    image_scores = []
    image_labels = []
    all_pixel_scores = []
    all_pixel_labels = []
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    print(f"Evaluating {args.category} (Improved PatchCore)...")
    with torch.no_grad():
        for i, (images, label, masks, names) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            
            # Extract features (multi-scale)
            features = model(images)
            
            # Aggregate spatial context (3x3 average pooling)
            for j, f in enumerate(features):
                f_pooled = F.avg_pool2d(f, 3, stride=1, padding=1)
                features[j] = f_pooled
            
            # Combine features (Upsample Layer 3 to Layer 2 resolution)
            f1, f2 = features
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            combined = torch.cat([f1, f2_up], dim=1) 
            
            # L2 Normalize features along the channel dimension
            combined = F.normalize(combined, p=2, dim=1)
            
            h_feat, w_feat = combined.shape[-2:]
            test_features = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1]).cpu().numpy()
            
            # Find distances to the 9 nearest normal neighbors
            distances, _ = knn.kneighbors(test_features) # [H*W, 9]
            
            # Robust Image Score: Weighted Top-0.1% of patch scores (Finer focus for AD 2)
            patch_scores = distances[:, 0]
            top_k = max(1, int(len(patch_scores) * 0.001))
            top_scores = np.sort(patch_scores)[-top_k:]
            
            # Find the index of the absolute max for weighting
            max_idx = np.argmax(patch_scores)
            weights = 1 - (np.exp(patch_scores[max_idx]) / np.sum(np.exp(distances[max_idx, :])))
            
            image_score = weights * np.mean(top_scores)
            
            # Reshape to 2D anomaly map for pixel metrics
            anomaly_map = patch_scores.reshape(h_feat, w_feat)
            
            # Upsample anomaly map
            anomaly_map_resized = cv2.resize(anomaly_map, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
            anomaly_map_resized = cv2.GaussianBlur(anomaly_map_resized, (3, 3), 0)
            
            image_scores.append(image_score)
            image_labels.append(label.item())
            
            # Pixel-level labels
            mask_np = masks[0, 0].cpu().numpy()
            mask_np = (mask_np > 0.5).astype(int)
            all_pixel_scores.append(anomaly_map_resized.flatten())
            all_pixel_labels.append(mask_np.flatten())
            
            # Visualization (First 20)
            if i < 20:
                img_np = denormalize(images[0]).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                heatmap, overlay = get_heatmap(anomaly_map_resized, img_np)
                save_path = os.path.join(args.result_dir, f"{args.category}_{i}_{names[0]}")
                plot_results(images[0], masks[0], anomaly_map_resized, overlay, save_path=save_path)
            
    # --- Metrics ---
    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)
    img_auc = roc_auc_score(image_labels, image_scores)
    
    # Calculate optimal threshold for F1 score (Post-hoc)
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(image_labels, image_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_post = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    best_f1_post = f1_scores[best_f1_idx]
    
    # Realistic Threshold: 99th percentile of normal scores (as per MVTec AD 2 guidelines)
    normal_scores = image_scores[image_labels == 0]
    if len(normal_scores) > 0:
        realistic_threshold = np.percentile(normal_scores, 99)
    else:
        realistic_threshold = best_threshold_post
    
    # Predictions at realistic threshold
    preds = (image_scores >= realistic_threshold).astype(int)
    from sklearn.metrics import f1_score as f1_metric
    realistic_f1 = f1_metric(image_labels, preds)
    
    # --- Visualization Pass ---
    print("Saving visualizations...")
    # We'll collect the top/wrong samples in the main loop next time, 
    # but for now, let's just make sure the results directory exists and we save something.
    os.makedirs(args.result_dir, exist_ok=True)
    
    # FP and FN
    fp_indices = np.where((preds == 1) & (image_labels == 0))[0]
    fn_indices = np.where((preds == 0) & (image_labels == 1))[0]
    
    fp = len(fp_indices)
    fn = len(fn_indices)
    
    # Get names from dataloader (requires keeping track of names)
    all_names = []
    for _, _, _, names in dataloader:
        all_names.extend(names)
    
    fp_names = [all_names[i] for i in fp_indices]
    fn_names = [all_names[i] for i in fn_indices]
    
    # Calculate AU-PRO (Per-region overlap) before flattening/concatenating
    pixel_aupro_30 = calculate_aupro(all_pixel_scores, all_pixel_labels, fpr_limit=0.3)
    pixel_aupro_05 = calculate_aupro(all_pixel_scores, all_pixel_labels, fpr_limit=0.05)
    
    all_pixel_scores = np.concatenate(all_pixel_scores)
    all_pixel_labels = np.concatenate(all_pixel_labels)
    pixel_auc = roc_auc_score(all_pixel_labels, all_pixel_scores)
    
    print(f"\n{args.category} Final Metrics (Frozen PatchCore):")
    print(f"  > Image-level AUROC: {img_auc:.4f}")
    print(f"  > [Post-hoc] Best F1: {best_f1_post:.4f} (at threshold {best_threshold_post:.4f})")
    print(f"  > [Realistic] F1:     {realistic_f1:.4f} (at threshold {realistic_threshold:.4f})")
    print(f"  > False Positives:   {fp} {fp_names} (Total normal: {np.sum(image_labels==0)})")
    print(f"  > False Negatives:   {fn} {fn_names} (Total anomaly: {np.sum(image_labels==1)})")
    print(f"  > Pixel-level AUROC: {pixel_auc:.4f} (Localization)")
    print(f"  > Pixel-level AU-PRO (0.3): {pixel_aupro_30:.4f}")
    print(f"  > Pixel-level AU-PRO (0.05): {pixel_aupro_05:.4f}")
    
    with open(os.path.join(args.result_dir, "metrics.txt"), "a") as f:
        f.write(f"{args.category}: Image_AUC={img_auc:.4f}, Realistic_F1={realistic_f1:.4f}, AU-PRO_0.05={pixel_aupro_05:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Frozen Feature Matching for Anomaly Detection")
    parser.add_argument("--root_dir", type=str, default="mvtec_ad", help="Path to MVTec AD dataset")
    parser.add_argument("--category", type=str, default="bottle", help="Category to evaluate")
    parser.add_argument("--resolution", type=int, default=384, help="Image resolution")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory where memory bank is saved")
    parser.add_argument("--result_dir", type=str, default="results", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    evaluate(args)
