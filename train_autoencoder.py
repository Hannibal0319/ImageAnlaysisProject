import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloader
from model import CustomAutoencoder


def train_autoencoder(
    root_dir="mvtec_ad",
    category="bottle",
    resolution=256,
    batch_size=16,
    epochs=20,
    lr=1e-3,
    checkpoint_dir="checkpoints"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(
        root_dir=root_dir,
        category=category,
        split="train",
        batch_size=batch_size,
        resolution=resolution,
        shuffle=True
    )

    model = CustomAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, _, _, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(device)

            reconstructed = model(images)

            loss = loss_fn(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: loss = {avg_loss:.6f}")

    save_path = os.path.join(checkpoint_dir, "custom_autoencoder.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained autoencoder to {save_path}")


if __name__ == "__main__":
    train_autoencoder()