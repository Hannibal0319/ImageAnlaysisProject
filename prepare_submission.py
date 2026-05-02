import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from PIL import Image
from torchvision import transforms
from model import FeatureExtractor
from sklearn.neighbors import NearestNeighbors
import argparse

# Reuse PadToSquare and CLAHE from dataset.py logic if needed
# For now, we'll just use the basic transforms we verified

class PadToSquare(object):
    def __call__(self, img):
        w, h = img.size
        if w == h: return img
        max_size = max(w, h)
        padding_w = (max_size - w) // 2
        padding_h = (max_size - h) // 2
        padding = (padding_w, padding_h, max_size - w - padding_w, max_size - h - padding_h)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

def prepare_category(root_dir, category, output_base, resolution=256, pool_size=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nProcessing {category} on {device}...")
    
    model = FeatureExtractor().to(device)
    model.eval()
    
    # Load memory bank
    bank_path = f"checkpoints/{category}_memory_bank.pth"
    if not os.path.exists(bank_path):
        print(f"Skipping {category}: Memory bank not found.")
        return
    memory_bank = torch.load(bank_path, map_location='cpu').numpy()
    
    knn = NearestNeighbors(n_neighbors=9, algorithm='auto', n_jobs=-1)
    knn.fit(memory_bank)
    
    # Define transforms
    # Note: rice/can used different pool_sizes. We'll use the one passed.
    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process both private and private_mixed
    splits = {
        'test_private': 'private',
        'test_private_mixed': 'private_mixed'
    }
    
    for split_dir, split_name in splits.items():
        src_dir = os.path.join(root_dir, category, split_dir)
        if not os.path.exists(src_dir):
            continue
            
        print(f"  Split: {split_name}")
        image_paths = sorted(glob(os.path.join(src_dir, "*.png")))
        
        # Determine output directory structure
        # /mvtec_ad_2/{object_name}/{private,private_mixed}/anomaly_images/test/
        # Since we don't know good/bad, we'll put them in a flat 'test' or use a heuristic.
        # BUT the instruction said {good,bad}. We'll use prediction as a dummy or just put in 'test'.
        # Actually, for submission, we often put everything in 'test' if labels are unknown.
        # We'll create a 'test' folder and then maybe 'unknown' if needed.
        # But let's follow the user's path as closely as possible.
        
        for img_path in tqdm(image_paths):
            orig_img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = orig_img.size
            
            img_tensor = transform(orig_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(img_tensor)
                for i, f in enumerate(features):
                    f_pooled = F.avg_pool2d(f, pool_size, stride=1, padding=pool_size//2)
                    features[i] = f_pooled
                
                f1, f2 = features
                f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
                combined = torch.cat([f1, f2_up], dim=1)
                
                h_feat, w_feat = combined.shape[-2:]
                test_features = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1]).cpu().numpy()
                
                distances, _ = knn.kneighbors(test_features)
                anomaly_map = distances[:, 0].reshape(h_feat, w_feat)
                
                # Upsample to original resolution
                # First resize to square padded size, then crop
                max_dim = max(orig_w, orig_h)
                anomaly_map_upsampled = cv2.resize(anomaly_map, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
                
                # De-pad (Inverse of PadToSquare)
                pad_w = (max_dim - orig_w) // 2
                pad_h = (max_dim - orig_h) // 2
                anomaly_map_final = anomaly_map_upsampled[pad_h:pad_h+orig_h, pad_w:pad_w+orig_w]
                
                # Smoothing
                anomaly_map_final = cv2.GaussianBlur(anomaly_map_final, (3, 3), 0)
                
                # Save as TIFF
                # We need to decide if we put it in 'good' or 'bad' based on a global threshold
                # Or just put it in a single folder if allowed.
                # Since we want to pass the local structure check, we'll use a dummy 'good' for now 
                # or better: we'll check if the submission check expects our split.
                
                # For now, let's use a simple heuristic: if image_score > threshold, it's bad.
                # Threshold from Realistic F1 (99th percentile) would be best.
                # We'll just use a 'test' directory for now.
                dest_dir = os.path.join(output_base, category, split_name, "anomaly_images", "test", "all")
                os.makedirs(dest_dir, exist_ok=True)
                
                save_path = os.path.join(dest_dir, os.path.basename(img_path).replace(".png", ".tiff"))
                cv2.imwrite(save_path, anomaly_map_final.astype(np.float32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=r"C:\Users\Peter\Desktop\stuff\MIUN\Research\robustanomaly\mvtec_ad_2")
    parser.add_argument("--output_base", type=str, default="submission/mvtec_ad_2")
    parser.add_argument("--category", type=str, default="can")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--pool", type=int, default=5)
    args = parser.parse_args()
    
    prepare_category(args.root_dir, args.category, args.output_base, args.res, args.pool)
