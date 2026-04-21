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
from dataset import get_dataloader
from model import get_model
from utils import denormalize


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
    
    # Dataloader with hardware optimizations
    dataloader = get_dataloader(
        args.root_dir, 
        args.category, 
        split='train', 
        batch_size=args.batch_size, 
        resolution=args.resolution,
        num_workers=args.num_workers,
        pin_memory=True
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Create checkpoints directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (images, labels, masks, names) in enumerate(pbar):
            # Move to device and memory format
            images = images.to(device, memory_format=torch.channels_last if device.type == 'cuda' else None)
            
            # Forward with Automatic Mixed Precision
            try:
                with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                    reconstructions = model(images)
                    loss = criterion(reconstructions, images)
            except RuntimeError as e:
                if "triton" in str(e).lower() and is_compiled:
                    print("\n" + "!" * 50)
                    print("ERROR: Lazy compilation failed due to missing Triton.")
                    print("Falling back to standard model for the rest of the run.")
                    print("!" * 50 + "\n")
                    # Break out and fallback
                    model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    is_compiled = False
                    # Retry this batch
                    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                        reconstructions = model(images)
                        loss = criterion(reconstructions, images)
                else:
                    raise e
            
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
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', args.lr, epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        
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
