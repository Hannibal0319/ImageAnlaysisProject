import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataset import get_dataloader
from model import FeatureExtractor
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import argparse

def analyze_category(category="can"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = FeatureExtractor().to(device)
    model.eval()
    
    bank_path = f"checkpoints/{category}_memory_bank.pth"
    if not os.path.exists(bank_path):
        print(f"Memory bank {bank_path} not found. Please train first.")
        return
        
    memory_bank = torch.load(bank_path, map_location='cpu').numpy()
    
    knn = NearestNeighbors(n_neighbors=9, algorithm='auto', n_jobs=-1)
    knn.fit(memory_bank)
    
    dataloader = get_dataloader(r"C:\Users\Peter\Desktop\stuff\MIUN\Research\robustanomaly\mvtec_ad_2", category, split='test', batch_size=1, resolution=384)
    
    results = []
    
    print(f"Analyzing {category} conditions...")
    with torch.no_grad():
        for i, (images, label, masks, names) in enumerate(tqdm(dataloader)):
            features = model(images.to(device))
            for j, f in enumerate(features):
                f_pooled = F.avg_pool2d(f, 3, stride=1, padding=1)
                features[j] = f_pooled
            
            f1, f2 = features
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            combined = torch.cat([f1, f2_up], dim=1)
            
            # Subsample evaluation patches for speed
            test_features = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1]).cpu().numpy()
            test_features = test_features[::10] # Evaluation subsampling
            
            distances, _ = knn.kneighbors(test_features)
            patch_scores = distances[:, 0]
            
            # Image score: Top-0.1% mean
            top_k = max(1, int(len(patch_scores) * 0.001))
            top_scores = np.sort(patch_scores)[-top_k:]
            image_score = np.mean(top_scores)
            
            # Determine condition from name (e.g., 001_overexposed.png)
            name = names[0]
            condition = "regular"
            if "overexposed" in name: condition = "overexposed"
            elif "underexposed" in name: condition = "underexposed"
            elif "shift" in name: condition = "shift"
            
            results.append({
                'name': name,
                'label': label.item(),
                'score': image_score,
                'condition': condition
            })

    df = pd.DataFrame(results)
    print("\nScore Analysis by Condition:")
    summary = df.groupby(['condition', 'label'])['score'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(summary)
    
    # Check overlap
    for cond in df['condition'].unique():
        good_scores = df[(df['condition'] == cond) & (df['label'] == 0)]['score']
        bad_scores = df[(df['condition'] == cond) & (df['label'] == 1)]['score']
        if not good_scores.empty and not bad_scores.empty:
            print(f"\nCondition: {cond}")
            print(f"  Max Good: {good_scores.max():.4f}")
            print(f"  Min Bad:  {bad_scores.min():.4f}")
            if good_scores.max() < bad_scores.min():
                print("  🟢 Perfectly separable within this condition")
            else:
                print("  🔴 Overlap detected")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="can")
    args = parser.parse_index_args() if hasattr(parser, 'parse_index_args') else parser.parse_args()
    analyze_category(args.category)
