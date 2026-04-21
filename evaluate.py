import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
from dataset import get_dataloader
from model import get_model
from utils import get_heatmap, plot_results, calculate_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloader
    dataloader = get_dataloader(args.root_dir, args.category, split='test', batch_size=1, shuffle=False, resolution=args.resolution)
    
    # Model
    model = get_model().to(device)
    model_path = os.path.join(args.checkpoint_dir, f"{args.category}_autoencoder.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    anomaly_scores = []
    labels = []
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    print(f"Evaluating {args.category}...")
    with torch.no_grad():
        for i, (images, label, masks, names) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            reconstructions = model(images)
            
            # Reconstruction error for heatmap
            diff, overlay = get_heatmap(images, reconstructions)
            
            # Anomaly score: mean pixel-wise error
            score = diff.mean()
            anomaly_scores.append(score)
            labels.append(label.item())
            
            # Save visual results for a few samples
            if i < 20: 
                save_path = os.path.join(args.result_dir, f"{args.category}_{i}_{names[0]}")
                plot_results(images, reconstructions, masks, diff, overlay, save_path=save_path)
                
    # Binary classification based on anomaly score threshold
    threshold = np.mean(anomaly_scores)
    preds = (np.array(anomaly_scores) > threshold).astype(int)
    true_labels = np.array(labels)
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, zero_division=0)
    rec = recall_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds, zero_division=0)
    print(f"\n{args.category} Test Metrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    
    # Save metrics to a file
    with open(os.path.join(args.result_dir, "metrics.txt"), "a") as f:
        f.write(f"{args.category}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN Autoencoder for Anomaly Detection")
    parser.add_argument("--root_dir", type=str, default="mvtec_ad", help="Path to MVTec AD dataset")
    parser.add_argument("--category", type=str, default="bottle", help="Category to evaluate")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory where models are saved")
    parser.add_argument("--result_dir", type=str, default="results", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    evaluate(args)
