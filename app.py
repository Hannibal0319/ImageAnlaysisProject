import gradio as gr
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
from utils import denormalize, get_heatmap
import cv2

# Global settings
RESOLUTION = 256
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
    model = get_model().to(device)
    model_path = os.path.join(CHECKPOINT_DIR, f"{category}_autoencoder.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[category] = model
        return model
    else:
        return None

def predict(image, category):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_category_model(category)
    
    if model is None:
        return None, None, f"Model for '{category}' not found in {CHECKPOINT_DIR}. Please train it first.", 0
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        recon_tensor = model(img_tensor)
        
    # Get heatmap and localization
    diff, overlay = get_heatmap(img_tensor, recon_tensor)
    
    # Reconstructed image for display
    recon_img = denormalize(recon_tensor[0]).permute(1, 2, 0).cpu().detach().numpy()
    recon_img = np.clip(recon_img * 255, 0, 255).astype(np.uint8)
    
    # Anomaly Score
    score = diff.mean()
    threshold = 5.0 # This should ideally be calibrated per category
    status = "DEFECTIVE" if score > threshold else "NORMAL"
    
    return recon_img, overlay, f"Status: {status} (Score: {score:.2f})", score

# UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Object Defect Detection using CNN Autoencoder")
    gr.Markdown("Identify and localize defects in industrial images using unsupervised learning.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            category_selector = gr.Dropdown(choices=CATEGORIES, value="bottle", label="Select Category")
            detect_btn = gr.Button("Detect Anomaly", variant="primary")
            
        with gr.Column():
            with gr.Row():
                recon_out = gr.Image(label="Reconstruction")
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
    demo.launch(share=False)
