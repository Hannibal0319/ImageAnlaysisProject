# Anomaly Detection with CNN Autoencoder

This project implements an unsupervised anomaly detection system using a Convolutional Autoencoder on the MVTec AD dataset.

## Features
- **CNN Autoencoder**: A reconstructive model that learns to compress and restore normal images.
- **Synthetic Anomaly Generation**: Trains the model to identify and localize defects using artificially generated anomalies (noise patches, intensity shifts).
- **Overfitting Monitoring**: Real-time tracking of Training vs. Validation MSE and Accuracy using a 90/10 training split.
- **GPU Optimization**: Full CUDA support for high-performance training on NVIDIA GPUs (e.g., RTX 4060).
- **TensorBoard Integration**: Visualize training curves and localized defect heatmaps.

## Architecture
- **Encoder**: 5 layers of convolutions with BatchNorm, ReLU, and Dropout (0.2).
- **Decoder**: 5 layers of transposed convolutions to reconstruct the original image.
- **Classification Head**: Uses reconstruction error as a feature for binary classification (Normal vs. Anomaly).

## Setup
1. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio numpy scikit-learn tqdm opencv-python matplotlib
   ```
2. Download the MVTec AD dataset and place it in the project root.

## Usage

### Training
Train the model on a specific category (e.g., 'bottle'):
```bash
py -3.12 train.py --category bottle --epochs 50 --batch_size 16
```
The script will:
- Save the final model and the **best model** (based on Val Accuracy) in the `checkpoints/` directory.
- Log metrics to `runs/` for TensorBoard.

### Evaluation
Evaluate the model on the test set:
```bash
py -3.12 evaluate.py --category bottle
```
This produces Accuracy, Precision, Recall, and F1 metrics and saves visual results in the `results/` directory.

### Monitoring
Launch TensorBoard to view progress:
```bash
tensorboard --logdir runs
```

## Metrics
- **MSE**: Reconstruction error (minimized during training).
- **Accuracy/F1**: Binary classification performance on synthetic (training) and real (test) anomalies.
