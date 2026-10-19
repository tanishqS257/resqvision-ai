import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.localization import build_localization_model
from src.data.dataset import BuildingLocalizationDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Hyperparameters tuned for 4GB VRAM (RTX 2050)
    batch_size = 4
    num_epochs = 5
    learning_rate = 1e-4

    # 2. Dataset and train/val split (80% train, 20% val)
    # Using a 1,000-sample subset for a rapid first milestone run
    print("Loading dataset...")
    full_dataset = BuildingLocalizationDataset(data_dir=r"E:\train", image_size=(512, 512), max_samples=1000)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 3. Model, Loss, Optimizer
    model = build_localization_model(device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 4. Training loop
    os.makedirs("artifacts", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} finished — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "artifacts/localization_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    print("Training finished successfully.")

if __name__ == "__main__":
    train()