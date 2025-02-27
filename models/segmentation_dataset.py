import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform_img=None, transform_label=None):
        """
        root_dir: Path to Dataset/processed/TrainVal
        Assumes:
          - Color images in: root_dir/color
          - Label images in: root_dir/label
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "color"
        self.label_dir = self.root_dir / "label"
        self.transform_img = transform_img
        self.transform_label = transform_label

        # List all image files in img_dir (skip the label subfolder)
        self.image_files = [f for f in self.img_dir.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        
        # Debug print
        print(f"Looking for images in {self.img_dir}")
        print(f"Found {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Assume label has the same stem with .png extension in label_dir
        label_path = self.label_dir / f"{img_path.stem}.png"
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # single-channel

        if self.transform_img:
            image = self.transform_img(image)
        else:
            image = transforms.ToTensor()(image)
        if self.transform_label:
            label = self.transform_label(label)
        else:
            label = torch.tensor(np.array(label), dtype=torch.long)
        return image, label
