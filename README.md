# Industrial Anomaly Detection (PatchCore-Lite)

This project implements a state-of-the-art unsupervised anomaly detection and localization system using **Feature-based Matching (PatchCore)** on the MVTec AD dataset.

## 🚀 Performance (Bottle Category)
- **Image-level AUROC**: **1.0000** (Perfect Classification)
- **Pixel-level AUROC**: **0.9822** (Precise Localization)
- **Precision**: **1.0000**

## ✨ Features
- **Zero-Training Architecture**: Leverages pre-trained ResNet-18 features. No backpropagation is required for new categories.
- **Scientific Evaluation**: Built-in support for **Pixel-AUROC**, the industry standard for measuring heatmap accuracy.
- **High-Resolution Heatmaps**: Generates localized anomaly maps without the limitations of standard Grad-CAM.
- **GPU Optimized**: Full CUDA support for faster feature extraction and nearest-neighbor search.

## 🏗️ Architecture
- **Backbone**: Pre-trained ResNet-18 (frozen).
- **Memory Bank**: Stores multi-scale patch embeddings from "normal" training data.
- **Anomaly Score**: Computed as the distance to the nearest neighbor in the normal feature space.
- **Localization**: Blinearly upsampled distance maps with Gaussian smoothing.

## 🛠️ Setup
1. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy scikit-learn tqdm opencv-python matplotlib
   ```
2. **Dataset**: Place the MVTec AD dataset in the `mvtec_ad/` directory.

## 📖 Usage

### 1. Build Memory Bank (The "Fingerprint")
Run this once for each category to collect its "normal" features:
```bash
py -3.12 train.py --category bottle --resolution 224
```
*Creates `checkpoints/bottle_memory_bank.pth`.*

### 2. Evaluate & Localize
Assess the test set and generate heatmaps:
```bash
py -3.12 evaluate.py --category bottle --resolution 224
```
*Computes Image and Pixel AUROC and saves results to the `results/` folder.*

### 3. Monitoring
Check `results/metrics.txt` for historical performance tracking and browse the saved heatmaps to verify localization.

## 📁 Project Structure
- `model.py`: Frozen ResNet-18 feature extractor.
- `train.py`: Feature collection and memory bank construction.
- `evaluate.py`: NN-based anomaly scoring and scientific metric calculation.
- `dataset.py`: Multi-threaded data loading with mask resolution handling.
- `utils.py`: Visualization and heatmap generation utilities.
