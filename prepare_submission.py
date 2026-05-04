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
import argparse
import tifffile
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), os.path.basename(path)

class PadToSquare(object):
    def __call__(self, img):
        w, h = img.size
        if w == h: return img
        max_size = max(w, h)
        padding_w = (max_size - w) // 2
        padding_h = (max_size - h) // 2
        padding = (padding_w, padding_h, max_size - w - padding_w, max_size - h - padding_h)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

def prepare_category(root_dir, category, output_base, resolution=256, pool_size=5, threshold=None, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nProcessing {category} on {device} (GPU Accelerated)...")
    
    model = FeatureExtractor().to(device)
    model.eval()
    
    # Load memory bank and move to GPU
    bank_path = f"checkpoints/{category}_memory_bank.pth"
    if not os.path.exists(bank_path):
        print(f"Skipping {category}: Memory bank not found.")
        return
        
    # We use weights_only=True if possible, but the earlier bank was saved with default
    memory_bank = torch.load(bank_path, map_location=device).to(torch.float32)
    
    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    splits = {
        'test_private': 'regular',
        'test_private_mixed': 'mixed'
    }
    
    for split_dir, suffix in splits.items():
        src_dir = os.path.join(root_dir, category, split_dir)
        if not os.path.exists(src_dir):
            continue
            
        print(f"  Split: {split_dir}")
        image_paths = sorted(glob(os.path.join(src_dir, "*.png")))
        dataset = SimpleDataset(image_paths, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
        
        raw_out_dir = os.path.join(output_base, "anomaly_images", category, split_dir)
        bin_out_dir = os.path.join(output_base, "anomaly_images_thresholded", category, split_dir)
        os.makedirs(raw_out_dir, exist_ok=True)
        os.makedirs(bin_out_dir, exist_ok=True)
        
        current_threshold = threshold if threshold is not None else 4.0

        for batch_idx, (imgs, filenames) in enumerate(tqdm(dataloader)):
            imgs = imgs.to(device)
            
            with torch.no_grad():
                features = model(imgs)
                for i, f in enumerate(features):
                    f_pooled = F.avg_pool2d(f, pool_size, stride=1, padding=pool_size//2)
                    features[i] = f_pooled
                
                f1, f2 = features
                f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
                combined = torch.cat([f1, f2_up], dim=1) 
                
                B, C, H, W = combined.shape
                # Flatten spatial dims to [B*H*W, C]
                test_features = combined.permute(0, 2, 3, 1).reshape(-1, C)
                
                # GPU Distance Calculation
                # Using squared Euclidean distance for speed then sqrt
                # d(x,y)^2 = |x|^2 + |y|^2 - 2xy
                dist_matrix = torch.cdist(test_features, memory_bank, p=2.0)
                
                # Nearest neighbor distance
                patch_scores, _ = torch.min(dist_matrix, dim=1)
                patch_scores = patch_scores.reshape(B, H, W)
                
                # Transfer to CPU for saving
                patch_scores_np = patch_scores.cpu().numpy()
                
                for i in range(len(filenames)):
                    # Get original image for correct size/cropping
                    # Note: We re-open just to get size, could be optimized further by pre-scanning
                    img_full_path = os.path.join(src_dir, filenames[i])
                    with Image.open(img_full_path) as temp_img:
                        orig_w, orig_h = temp_img.size
                    
                    max_dim = max(orig_w, orig_h)
                    score_map = patch_scores_np[i]
                    
                    # Upsample and crop
                    anomaly_map_upsampled = cv2.resize(score_map, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
                    pad_w = (max_dim - orig_w) // 2
                    pad_h = (max_dim - orig_h) // 2
                    anomaly_map_final = anomaly_map_upsampled[pad_h:pad_h+orig_h, pad_w:pad_w+orig_w]
                    anomaly_map_final = cv2.GaussianBlur(anomaly_map_final, (3, 3), 0)
                    
                    # Official naming: 000_regular.tiff
                    global_idx = batch_idx * batch_size + i
                    base_name = f"{str(global_idx).zfill(3)}_{suffix}"
                    
                    # Save TIFF (float16)
                    raw_path = os.path.join(raw_out_dir, f"{base_name}.tiff")
                    tifffile.imwrite(raw_path, anomaly_map_final.astype(np.float16))
                    
                    # Save Binary PNG (0, 255)
                    bin_mask = (anomaly_map_final > current_threshold).astype(np.uint8) * 255
                    bin_path = os.path.join(bin_out_dir, f"{base_name}.png")
                    cv2.imwrite(bin_path, bin_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=r"C:\Users\Peter\Desktop\stuff\MIUN\Research\robustanomaly\mvtec_ad_2")
    parser.add_argument("--output_base", type=str, default="submission")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--pool", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    categories = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
    
    if args.category == "all":
        for cat in categories:
            prepare_category(args.root_dir, cat, args.output_base, args.res, args.pool, args.threshold, args.batch_size)
    else:
        prepare_category(args.root_dir, args.category, args.output_base, args.res, args.pool, args.threshold, args.batch_size)
