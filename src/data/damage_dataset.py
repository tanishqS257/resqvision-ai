import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

class DamageClassificationDataset(Dataset):
    def __init__(self, data_dir=r"E:\train", image_size=(512, 512), max_samples=None):
        self.image_dir = os.path.join(data_dir, "images")
        self.target_dir = os.path.join(data_dir, "targets")
        self.image_size = image_size

        all_files = os.listdir(self.image_dir)
        self.post_files = sorted([f for f in all_files if f.endswith("_post_disaster.png")])

        if max_samples:
            self.post_files = self.post_files[:max_samples]

    def __len__(self):
        return len(self.post_files)

    def __getitem__(self, idx):
        post_name = self.post_files[idx]
        pre_name = post_name.replace("_post_disaster.png", "_pre_disaster.png")
        target_name = post_name.replace("_post_disaster.png", "_post_disaster_target.png")

        post_path = os.path.join(self.image_dir, post_name)
        pre_path = os.path.join(self.image_dir, pre_name)
        target_path = os.path.join(self.target_dir, target_name)

        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")
        target_img = Image.open(target_path).convert("L")

        pre_img = pre_img.resize(self.image_size, Image.Resampling.BILINEAR)
        post_img = post_img.resize(self.image_size, Image.Resampling.BILINEAR)
        target_img = target_img.resize(self.image_size, Image.Resampling.NEAREST)

        pre_tensor = TF.to_tensor(pre_img)
        post_tensor = TF.to_tensor(post_img)
        
        target_np = np.array(target_img)
        # Any value outside [0, 4] is mapped to 0 (background)
        target_np[target_np > 4] = 0
        target_tensor = torch.from_numpy(target_np).long()

        return pre_tensor, post_tensor, target_tensor