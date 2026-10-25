import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

class BuildingLocalizationDataset(Dataset):
    def __init__(self, data_dir=r"E:\train", image_size=(512, 512), max_samples=None):
        self.image_dir = os.path.join(data_dir, "images")
        self.target_dir = os.path.join(data_dir, "targets")
        self.image_size = image_size

        # Localization only trains on pre-disaster imagery
        all_files = os.listdir(self.image_dir)
        self.image_files = sorted([f for f in all_files if f.endswith("_pre_disaster.png")])
        
        if max_samples:
            self.image_files = self.image_files[:max_samples]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        target_name = img_name.replace("_pre_disaster.png", "_pre_disaster_target.png")

        img_path = os.path.join(self.image_dir, img_name)
        target_path = os.path.join(self.target_dir, target_name)

        # 1. Load pre-disaster image and pre-rendered target mask
        image = Image.open(img_path).convert("RGB")
        target = Image.open(target_path).convert("L")

        # 2. Resize to 512x512 to accommodate the RTX 2050's 4GB VRAM
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        target = target.resize(self.image_size, Image.Resampling.NEAREST)

        # 3. Convert image to normalized tensor (0.0 to 1.0)
        image_tensor = TF.to_tensor(image)

        # 4. Convert target to a binary mask: 1 for building footprints, 0 for background
        target_np = np.array(target)
        mask = (target_np > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor