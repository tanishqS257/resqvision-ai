import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.damage_classifier import build_damage_model
from src.data.damage_dataset import DamageClassificationDataset

def train_damage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Damage Classifier on: {device} ({torch.cuda.get_device_name(0)})")

    batch_size = 4
    num_epochs = 5
    learning_rate = 1e-4

    print("Loading damage dataset...")
    full_dataset = DamageClassificationDataset(data_dir=r"E:\train", image_size=(512, 512), max_samples=1000)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_damage_model(device=device)

    # Class weights to balance underrepresented severe damage:
    # 0: background (low weight), 1: no-damage, 2: minor, 3: major, 4: destroyed
    class_weights = torch.tensor([0.2, 1.0, 2.0, 3.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for pre_imgs, post_imgs, targets in pbar:
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(pre_imgs, post_imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pre_imgs, post_imgs, targets in val_loader:
                pre_imgs = pre_imgs.to(device)
                post_imgs = post_imgs.to(device)
                targets = targets.to(device)
                outputs = model(pre_imgs, post_imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "artifacts/damage_classifier_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best damage checkpoint to {save_path}")

    print("Damage classifier training finished successfully.")

if __name__ == "__main__":
    train_damage()