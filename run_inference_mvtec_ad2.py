import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import tifffile
from tqdm import tqdm
from glob import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import get_model
from sklearn.neighbors import NearestNeighbors
import argparse

class SimpleDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), os.path.basename(path), path

class PadToSquare(object):
    def __call__(self, img):
        w, h = img.size
        if w == h: return img
        max_size = max(w, h)
        padding_w = (max_size - w) // 2
        padding_h = (max_size - h) // 2
        padding = (padding_w, padding_h, max_size - w - padding_w, max_size - h - padding_h)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

def torch_nn_search(test_features, memory_bank, device, chunk_size=4096, k=9):
    """GPU-accelerated NN search using torch.cdist for K neighbors."""
    num_queries = test_features.shape[0]
    all_dists = torch.zeros(num_queries, k, device='cpu')
    
    # Process in chunks to avoid OOM on GPU
    for i in range(0, num_queries, chunk_size):
        chunk = test_features[i:i+chunk_size].to(device)
        # cdist: (1, chunk_size, C) vs (1, bank_size, C) -> (1, chunk_size, bank_size)
        dists = torch.cdist(chunk.unsqueeze(0), memory_bank.unsqueeze(0), p=2).squeeze(0)
        # Get K smallest distances
        vals, _ = torch.topk(dists, k, dim=1, largest=False)
        all_dists[i:i+chunk_size] = vals.cpu()
        
    return all_dists.numpy()

def calculate_threshold(cat, model, memory_bank, transform, device, args):
    """Calculates threshold from validation set using evaluate.py logic (99th percentile of image scores)."""
    val_dir = os.path.join(args.root_dir, cat, "validation", "good")
    if not os.path.exists(val_dir):
        print(f"    Warning: Validation dir not found for {cat}. Using default threshold.")
        return args.threshold
        
    val_paths = sorted(glob(os.path.join(val_dir, "*.png")))
    if not val_paths:
        return args.threshold
        
    val_ds = SimpleDataset(val_paths, transform)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    
    val_image_scores = []
    print(f"    Calculating threshold for {cat} (Matching evaluate.py) from {len(val_paths)} images...")
    
    model.eval()
    with torch.no_grad():
        for imgs, filenames, full_paths in val_dl:
            imgs = imgs.to(device)
            features = model(imgs)
            for i, f in enumerate(features):
                features[i] = F.avg_pool2d(f, 5, stride=1, padding=2)
            
            f1, f2 = features
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            combined = torch.cat([f1, f2_up], dim=1)
            
            B, C, H, W = combined.shape
            # For each image in batch
            for b in range(B):
                test_features = combined[b:b+1].permute(0, 2, 3, 1).reshape(-1, C)
                # K=9 search
                distances = torch_nn_search(test_features, memory_bank, device, k=9)
                patch_scores = distances[:, 0]
                
                # Image Intensity Masking (Trick from evaluate.py)
                img_intensity = torch.mean(imgs[b], dim=0).cpu().numpy()
                intensity_mask = cv2.resize(img_intensity, (H, W), interpolation=cv2.INTER_LINEAR)
                foreground_mask = (intensity_mask > -1.5).flatten()
                
                # Filter scores
                masked_patch_scores = patch_scores[foreground_mask]
                masked_distances = distances[foreground_mask]
                
                if len(masked_patch_scores) == 0:
                    val_image_scores.append(0.0)
                    continue
                
                # Top 1% average (Trick from evaluate.py)
                top_k = max(1, int(len(masked_patch_scores) * 0.01))
                top_scores = np.sort(masked_patch_scores)[-top_k:]
                
                # Softmax Weighting (Trick from evaluate.py)
                max_idx = np.argmax(masked_patch_scores)
                weight = 1 - (np.exp(masked_patch_scores[max_idx]) / np.sum(np.exp(masked_distances[max_idx, :])))
                val_image_scores.append(weight * np.mean(top_scores))
            
    val_image_scores = np.array(val_image_scores)
    # Threshold is 99th percentile of normal validation images (Matching evaluate.py)
    threshold = np.percentile(val_image_scores, 99) if len(val_image_scores) > 0 else args.threshold
    print(f"    New threshold for {cat}: {threshold:.4f} (99th percentile of validation)")
    return threshold

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    categories = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
    if args.category != "all":
        categories = [args.category]
        
    model = get_model().to(device)
    model.eval()
    
    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    splits = {'test_private': 'private', 'test_private_mixed': 'private mixed'}
    
    for cat in categories:
        print(f"\nProcessing category: {cat}")
        bank_path = os.path.join(args.checkpoint_dir, f"{cat}_memory_bank.pth")
        if not os.path.exists(bank_path):
            continue
            
        memory_bank = torch.load(bank_path, map_location='cpu').to(device)
        current_threshold = calculate_threshold(cat, model, memory_bank, transform, device, args)
        
        for split_dir, split_name in splits.items():
            src_dir = os.path.join(args.root_dir, cat, split_dir)
            if not os.path.exists(src_dir): continue
                
            image_paths = sorted(glob(os.path.join(src_dir, "*.png")))
            dataset = SimpleDataset(image_paths, transform)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2)
            
            for batch_idx, (imgs, filenames, full_paths) in enumerate(tqdm(dataloader, desc=f"    Inference {cat}/{split_name}")):
                imgs = imgs.to(device)
                with torch.no_grad():
                    features = model(imgs)
                    for i, f in enumerate(features):
                        features[i] = F.avg_pool2d(f, 5, stride=1, padding=2)
                    
                    f1, f2 = features
                    f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
                    combined = torch.cat([f1, f2_up], dim=1) 
                    
                    B, C, H, W = combined.shape
                    
                for b in range(B):
                    test_features = combined[b:b+1].permute(0, 2, 3, 1).reshape(-1, C)
                    distances = torch_nn_search(test_features, memory_bank, device, k=9)
                    patch_scores = distances[:, 0]
                    
                    # Image Intensity Masking
                    img_intensity = torch.mean(imgs[b], dim=0).cpu().numpy()
                    intensity_mask_small = cv2.resize(img_intensity, (H, W), interpolation=cv2.INTER_LINEAR)
                    foreground_mask_small = (intensity_mask_small > -1.5).flatten()
                    
                    # Robust Image Score Calculation
                    masked_patch_scores = patch_scores[foreground_mask_small]
                    masked_distances = distances[foreground_mask_small]
                    
                    if len(masked_patch_scores) > 0:
                        top_k = max(1, int(len(masked_patch_scores) * 0.01))
                        top_scores = np.sort(masked_patch_scores)[-top_k:]
                        max_idx = np.argmax(masked_patch_scores)
                        weight = 1 - (np.exp(masked_patch_scores[max_idx]) / np.sum(np.exp(masked_distances[max_idx, :])))
                        image_score = weight * np.mean(top_scores)
                    else:
                        image_score = 0.0
                    
                    classification = "bad" if image_score > current_threshold else "good"
                    
                    # Final Map Preparation
                    score_map = patch_scores.reshape(H, W)
                    with Image.open(full_paths[b]) as temp_img:
                        orig_w, orig_h = temp_img.size
                    
                    max_dim = max(orig_w, orig_h)
                    anomaly_map_upsampled = cv2.resize(score_map, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
                    pad_w = (max_dim - orig_w) // 2
                    pad_h = (max_dim - orig_h) // 2
                    anomaly_map_final = anomaly_map_upsampled[pad_h:pad_h+orig_h, pad_w:pad_w+orig_w]
                    anomaly_map_final = cv2.GaussianBlur(anomaly_map_final, (3, 3), 0)
                    
                    # Apply original resolution intensity mask to final map
                    intensity_mask_orig = cv2.resize(img_intensity, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    foreground_mask_orig = (intensity_mask_orig > -1.5).astype(np.float32)
                    anomaly_map_final *= foreground_mask_orig
                    
                    # Save outputs
                    tiff_name = filenames[b].replace(".png", ".tiff")
                    
                    # 1. Custom structure
                    custom_out_dir = os.path.join(args.output_base, "mvtec ad 2", cat, split_name, "anomaly images", "test", classification)
                    os.makedirs(custom_out_dir, exist_ok=True)
                    tifffile.imwrite(os.path.join(custom_out_dir, tiff_name), anomaly_map_final.astype(np.float16))
                    
                    # 2. Official structure
                    sub_raw_dir = os.path.join(args.submission_dir, "anomaly_images", cat, split_dir)
                    sub_bin_dir = os.path.join(args.submission_dir, "anomaly_images_thresholded", cat, split_dir)
                    os.makedirs(sub_raw_dir, exist_ok=True)
                    os.makedirs(sub_bin_dir, exist_ok=True)
                    
                    tifffile.imwrite(os.path.join(sub_raw_dir, tiff_name), anomaly_map_final.astype(np.float16))
                    # Thresholding at pixel level for binary mask (Matching evaluate.py)
                    bin_mask = (anomaly_map_final > current_threshold).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(sub_bin_dir, filenames[b]), bin_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=r"C:\Users\Peter\Desktop\stuff\MIUN\Research\robustanomaly\mvtec_ad_2")
    parser.add_argument("--output_base", type=str, default=".") 
    parser.add_argument("--submission_dir", type=str, default="submission")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    run_inference(args)

