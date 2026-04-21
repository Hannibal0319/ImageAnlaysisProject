import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import multiprocessing
import platform
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import get_dataloader
from model import get_model
from utils import denormalize, generate_synthetic_anomaly


def is_triton_available():
    """Check if Triton is available for torch.compile backends."""
    try:
        import triton
        return True
    except ImportError:
        return False

def train(args):
    # Set CUDNN benchmark for fixed-size inputs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # TensorBoard setup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.category}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    
    # Dataloader with hardware optimizations and validation split
    full_dataset = get_dataloader(
        args.root_dir, 
        args.category, 
        split='train', 
        batch_size=args.batch_size, 
        resolution=args.resolution,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        get_dataset=True  # We'll add this flag to get_dataloader
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    
    # Model with memory format optimization
    model = get_model().to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    
    # Compile model for faster execution (PyTorch 2.0+)
    # Added robust check for Windows/Triton
    is_compiled = False
    if args.compile and hasattr(torch, 'compile'):
        triton_ready = is_triton_available()
        on_windows = platform.system() == "Windows"
        
        if on_windows and not triton_ready:
            print("\n" + "!" * 50)
            print("WARNING: torch.compile on Windows requires Triton.")
            print("Official Triton is not supported on native Windows.")
            print("Falling back to Eager Mode (standard execution).")
            print("!" * 50 + "\n")
        else:
            print("Compiling model (this may take a few minutes)...")
            try:
                # Use a dummy forward pass to catch lazy compilation errors early if possible
                compiled_model = torch.compile(model)
                is_compiled = True
                model = compiled_model
            except Exception as e:
                print(f"Warning: Model compilation failed ({e}). Proceeding without compile.")
                is_compiled = False
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Create checkpoints directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    model.train()
    best_val_acc = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (images, labels, masks, names) in enumerate(pbar):
            # Move to device and memory format
            images = images.to(device, memory_format=torch.channels_last if device.type == 'cuda' else None)
            
            # Generate synthetic anomalies for half of the batch
            corrupted_images, syn_labels = generate_synthetic_anomaly(images, p=0.5)
            
            # Forward pass
            reconstructions = model(corrupted_images)
            
            # Compute reconstruction loss (MSE) - compare against clean original images
            loss_mse = criterion(reconstructions, images)
            
            # Compute per-sample reconstruction error as anomaly score
            recon_error = torch.mean((reconstructions - corrupted_images) ** 2, dim=[1, 2, 3])
            
            # Binary classification loss (BCE with logits)
            bce_loss = nn.BCEWithLogitsLoss()(recon_error, syn_labels.float())
            
            # Combine losses
            loss = loss_mse + bce_loss

            
            # Backward and Optimize
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging to TensorBoard
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            pbar.set_postfix({'loss': loss.item()})
            
        # Log epoch-level metrics
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        
        # Validation pass
        model.eval()
        val_epoch_mse = 0
        val_scores = []
        val_labels = []
        with torch.no_grad():
            for val_images, val_labels_orig, val_masks, val_names in val_loader:
                val_images = val_images.to(device, memory_format=torch.channels_last if device.type == 'cuda' else None)
                
                # Add synthetic anomalies to the validation set
                val_corrupted, val_syn_labels = generate_synthetic_anomaly(val_images, p=0.5)
                
                val_reconstructions = model(val_corrupted)
                v_loss_mse = criterion(val_reconstructions, val_images)
                val_epoch_mse += v_loss_mse.item()
                
                # anomaly score
                v_recon_error = torch.mean((val_reconstructions - val_corrupted) ** 2, dim=[1, 2, 3])
                val_scores.extend(v_recon_error.cpu().numpy())
                val_labels.extend(val_syn_labels.cpu().numpy())
        
        avg_val_mse = val_epoch_mse / len(val_loader)
        
        # Binary classification metrics on synthetic anomalies
        val_threshold = np.mean(val_scores)
        val_preds = (np.array(val_scores) > val_threshold).astype(int)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        writer.add_scalar('Loss/val_mse', avg_val_mse, epoch)
        writer.add_scalar('Metrics/val_accuracy', val_acc, epoch)
        writer.add_scalar('Metrics/val_f1', val_f1, epoch)
        writer.add_scalar('Learning_Rate', args.lr, epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_loss:.4f}, Val MSE: {avg_val_mse:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model if accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.checkpoint_dir, f"{args.category}_best_autoencoder.pth")
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), best_path)
            print(f"  --> Saved new best model (Val Acc: {val_acc:.4f})")
            
        model.train()
        
        # Log visual results periodically
        if (epoch + 1) % args.log_interval == 0:
            with torch.no_grad():
                # Take first 4 images from the last batch
                n = min(args.batch_size, 4)
                orig = denormalize(images[:n])
                recon = denormalize(reconstructions[:n])
                
                # Create grids (n, 3, H, W)
                grid_orig = make_grid(orig, nrow=n)
                grid_recon = make_grid(recon, nrow=n)
                
                writer.add_image('Images/Original', grid_orig, epoch)
                writer.add_image('Images/Reconstruction', grid_recon, epoch)
        
    # Save final model
    save_path = os.path.join(args.checkpoint_dir, f"{args.category}_autoencoder.pth")
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(model_to_save.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    writer.close()

if __name__ == "__main__":
    # silencing warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # Fixed: Removed invalid '-1' setting. Use 'errors' or 'all' if needed, or leave unset.
    # os.environ['TORCH_LOGS'] = 'errors'

    parser = argparse.ArgumentParser(description="Train CNN Autoencoder (Optimized with Logging)")
    parser.add_argument("--root_dir", type=str, default="mvtec_ad", help="Path to MVTec AD dataset")
    parser.add_argument("--category", type=str, default="bottle", help="Category to train on")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--log_interval", type=int, default=5, help="Epoch interval for image logging")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count() // 2, help="Dataloader workers")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    
    args = parser.parse_args()
    train(args)
