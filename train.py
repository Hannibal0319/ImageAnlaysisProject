import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import multiprocessing
from dataset import get_dataloader
from model import get_model

def build_memory_bank(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloader (Training set only)
    dataloader = get_dataloader(
        args.root_dir, 
        args.category, 
        split='train', 
        batch_size=args.batch_size, 
        resolution=args.resolution,
        num_workers=args.num_workers
    )
    
    # Load frozen feature extractor
    model = get_model().to(device)
    model.eval()
    
    memory_bank = []
    
    print(f"Building frozen memory bank for {args.category}...")
    with torch.no_grad():
        for images, _, _, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # Extract multi-scale feature maps [Layer 2, Layer 3]
            features = model(images)
            
            # Aggregate spatial context (5x5 average pooling for granulated textures)
            for i, f in enumerate(features):
                f_pooled = nn.AvgPool2d(5, stride=1, padding=2)(f)
                features[i] = f_pooled
            
            # Combine features (Upsample Layer 3 to Layer 2 resolution)
            f1, f2 = features
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            combined = torch.cat([f1, f2_up], dim=1) 
            
            # Reshape to [B*H*W, C] (One vector per patch)
            combined = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1])
            
            # Subsample (every 10th patch for large AD 2 efficiency)
            combined = combined[::10]
            
            memory_bank.append(combined.cpu())

    # Concatenate all collected patches
    memory_bank = torch.cat(memory_bank, dim=0)
    
    # Create checkpoints directory and save
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, f"{args.category}_memory_bank.pth")
    
    
    torch.save(memory_bank, save_path)
    print(f"Memory bank built successfully! Shape: {memory_bank.shape}")
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Memory Bank for Feature Matching (Lite PatchCore)")
    parser.add_argument("--root_dir", type=str, default="mvtec_ad", help="Path to MVTec AD dataset")
    parser.add_argument("--category", type=str, default="bottle", help="Category to process")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for artifacts")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count() // 2, help="Dataloader workers")
    
    args = parser.parse_args()
    build_memory_bank(args)
