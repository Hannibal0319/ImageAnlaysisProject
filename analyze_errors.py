import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataset import get_dataloader
from model import FeatureExtractor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_curve
import cv2
import matplotlib.pyplot as plt

def visualize_error(image, mask, anomaly_map, save_path, title):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    mask = (mask[0].cpu().numpy() * 255).astype(np.uint8)
    
    # Anomaly map normalization for heatmap
    am_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
    heatmap = cv2.applyColorMap((am_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1); plt.imshow(image); plt.title("Original")
    plt.subplot(1, 4, 2); plt.imshow(mask, cmap='gray'); plt.title("GT Mask")
    plt.subplot(1, 4, 3); plt.imshow(heatmap); plt.title("Anomaly Map")
    plt.subplot(1, 4, 4); plt.imshow(overlay); plt.title(f"Overlay\nScore: {title}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_category(category="zipper"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().to(device)
    model.eval()
    
    bank_path = f"checkpoints/{category}_memory_bank.pth"
    memory_bank = torch.load(bank_path, map_location='cpu').numpy()
    
    knn = NearestNeighbors(n_neighbors=9, algorithm='auto', n_jobs=-1)
    knn.fit(memory_bank)
    
    dataloader = get_dataloader("mvtec_ad", category, split='test', batch_size=1, resolution=256)
    
    image_scores = []
    image_labels = []
    data_list = [] # Store raw data for later visualization
    
    print(f"Analyzing {category}...")
    with torch.no_grad():
        for i, (images, label, masks, names) in enumerate(tqdm(dataloader)):
            features = model(images.to(device))
            for j, f in enumerate(features):
                f_pooled = F.avg_pool2d(f, 3, stride=1, padding=1)
                features[j] = f_pooled
            
            f1, f2 = features
            f2_up = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            combined = torch.cat([f1, f2_up], dim=1)
            h_feat, w_feat = combined.shape[-2:]
            test_features = combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1]).cpu().numpy()
            
            distances, _ = knn.kneighbors(test_features)
            patch_scores = distances[:, 0]
            max_idx = np.argmax(patch_scores)
            weights = 1 - (np.exp(patch_scores[max_idx]) / np.sum(np.exp(distances[max_idx, :])))
            image_score = weights * patch_scores[max_idx]
            
            image_scores.append(image_score)
            image_labels.append(label.item())
            
            anomaly_map = patch_scores.reshape(h_feat, w_feat)
            anomaly_map_resized = cv2.resize(anomaly_map, (256, 256))
            anomaly_map_resized = cv2.GaussianBlur(anomaly_map_resized, (3, 3), 0)
            
            data_list.append({
                'image': images[0],
                'mask': masks[0],
                'anomaly_map': anomaly_map_resized,
                'score': image_score,
                'name': names[0]
            })

    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)
    precisions, recalls, thresholds = precision_recall_curve(image_labels, image_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    preds = (image_scores >= best_threshold).astype(int)
    
    os.makedirs("error_analysis", exist_ok=True)
    
    print(f"Threshold: {best_threshold:.4f}")
    
    for i, data in enumerate(data_list):
        label = image_labels[i]
        pred = preds[i]
        
        if label != pred:
            type_str = "FP" if pred == 1 else "FN"
            save_path = f"error_analysis/{type_str}_{data['name']}"
            visualize_error(data['image'], data['mask'], data['anomaly_map'], save_path, f"{data['score']:.4f}")
            print(f"Saved {type_str}: {data['name']} (Score: {data['score']:.4f})")

if __name__ == "__main__":
    analyze_category("zipper")
