import gradio as gr
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
from utils import denormalize, get_heatmap
import cv2
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

# Global settings
RESOLUTION = 224
CHECKPOINT_DIR = "checkpoints"
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model cache
models = {}

def load_category_model(category):
    if category in models:
        return models[category]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load frozen feature extractor
    model = get_model().to(device)
    model.eval()
    
    bank_path = os.path.join(CHECKPOINT_DIR, f"{category}_memory_bank.pth")
    
    if os.path.exists(bank_path):
        # Load memory bank
        memory_bank = torch.load(bank_path, map_location='cpu').numpy()
        
        # Fit Nearest Neighbors index
        knn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
        knn.fit(memory_bank)
        
        models[category] = (model, knn)
        return model, knn
    else:
        return None, None

def predict(image, category):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, knn = load_category_model(category)
    
    if model is None:
        return None, None, f"Memory bank for '{category}' not found in {CHECKPOINT_DIR}. Please build it first.", 0
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference (Feature Extraction)
    with torch.no_grad():
        features = model(img_tensor)
        
        # Aggregate spatial context (3x3 average pooling)
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
        distances, _ = knn.kneighbors(test_features)
        
        # Reshape to 2D anomaly map
        anomaly_map = distances.reshape(h_feat, w_feat)
        
        # Upsample anomaly map to original image resolution
        anomaly_map_resized = cv2.resize(anomaly_map, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
        anomaly_map_resized = cv2.GaussianBlur(anomaly_map_resized, (3, 3), 0)
        
    # Get original image for visualization
    img_np = denormalize(img_tensor[0]).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    # Get heatmap and localization
    heatmap, overlay = get_heatmap(anomaly_map_resized, img_np)
    
    # Anomaly Score
    score = np.max(anomaly_map_resized)
    threshold = 1.5 # Adjusted threshold for PatchCore
    status = "DEFECTIVE" if score > threshold else "NORMAL"
    
    return heatmap, overlay, f"Status: {status} (Score: {score:.2f})", score

# UI Layout
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Object Defect Detection using CNN Autoencoder")
    gr.Markdown("Identify and localize defects in industrial images using unsupervised learning.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            category_selector = gr.Dropdown(choices=CATEGORIES, value="bottle", label="Select Category")
            detect_btn = gr.Button("Detect Anomaly", variant="primary")
            
        with gr.Column():
            with gr.Row():
                recon_out = gr.Image(label="Anomaly Map (Raw)")
                overlay_out = gr.Image(label="Anomaly Heatmap")
            
            status_out = gr.Label(label="Classification Result")
            score_out = gr.Number(label="Anomaly Score")

    detect_btn.click(
        fn=predict,
        inputs=[input_img, category_selector],
        outputs=[recon_out, overlay_out, status_out, score_out]
    )
    
    gr.Examples(
        examples=[
            [os.path.join("mvtec_ad", "bottle", "test", "broken_large", "000.png"), "bottle"],
            [os.path.join("mvtec_ad", "bottle", "test", "good", "001.png"), "bottle"]
        ],
        inputs=[input_img, category_selector]
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
