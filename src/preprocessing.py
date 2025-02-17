import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import argparse
import random
from PIL import ImageFilter  # make sure to import this at the top
import shutil
import os

# --------------------------
# Resizing and Padding Classes
# --------------------------

class ResizeWithPadding:
    """
    For color images. Uses bilinear interpolation.
    """
    def __init__(self, target_size=512, padding_mode='mean', force_resize=True, resize_dims=(256, 256)):
        self.target_size = target_size
        self.force_resize = force_resize
        self.resize_dims = resize_dims
        assert padding_mode in ['mean', 'reflect', 'hybrid'], \
            "Padding mode must be 'mean', 'reflect', or 'hybrid'"
        self.padding_mode = padding_mode

    def resize_image(self, image):
        h, w = image.shape[:2]
        if self.force_resize:
            scale = self.resize_dims[0] / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale = self.resize_dims[0] / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def pad_image(self, image):
        h, w = image.shape[:2]
        # Determine target dimensions
        target_h, target_w = self.resize_dims if self.force_resize else (self.target_size, self.target_size)
        delta_w = max(0, target_w - w)
        delta_h = max(0, target_h - h)
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        if self.padding_mode == 'mean':
            mean_pixel = np.mean(image, axis=(0, 1), dtype=int)
            padded_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                              cv2.BORDER_CONSTANT, value=mean_pixel.tolist())
        elif self.padding_mode == 'reflect':
            padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
        else:  # hybrid
            reflected = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
            mean_pixel = np.mean(image, axis=(0, 1), dtype=int)
            padded_image = cv2.copyMakeBorder(reflected, top, bottom, left, right,
                                              cv2.BORDER_CONSTANT, value=mean_pixel.tolist())
        return padded_image

    def __call__(self, img):
        # Convert PIL image to numpy array, process, and convert back
        arr = np.array(img)
        resized = self.resize_image(arr)
        padded = self.pad_image(resized)
        return Image.fromarray(padded)

class ResizeWithPaddingLabel:
    """
    For labels/masks. Uses nearest-neighbor interpolation.
    Assumes labels are single-channel images.
    """
    def __init__(self, target_size=512, force_resize=False, resize_dims=(256, 256)):
        self.target_size = target_size
        self.force_resize = force_resize
        self.resize_dims = resize_dims

    def resize_image(self, image):
        h, w = image.shape[:2]
        if self.force_resize:
            scale = self.resize_dims[0] / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            scale = self.resize_dims[0] / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return resized_image

    def pad_image(self, image):
        h, w = image.shape[:2]
        target_h, target_w = self.resize_dims if self.force_resize else (self.target_size, self.target_size)
        delta_w = max(0, target_w - w)
        delta_h = max(0, target_h - h)
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left
        # For labels, use a constant value (e.g., 0 for background)
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return padded_image

    def __call__(self, label_img):
        arr = np.array(label_img)
        resized = self.resize_image(arr)
        padded = self.pad_image(resized)
        return Image.fromarray(padded)

# --------------------------
# Augmentation Function for Synchronized Transforms
# --------------------------

def elastic_transform_pair(image, label, alpha=34, sigma=4):
    """Apply elastic transformation to both image and label."""
    np_img = np.array(image)
    np_label = np.array(label)
    random_state = np.random.RandomState(None)
    shape = np_img.shape[:2]
    dx = (random_state.rand(*shape) * 2 - 1)
    dy = (random_state.rand(*shape) * 2 - 1)
    dx = cv2.GaussianBlur(dx, (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (17, 17), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    transformed_img = cv2.remap(np_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    transformed_label = cv2.remap(np_label, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(transformed_img), Image.fromarray(transformed_label)

def augment_pair(image, label, size=(256, 256), apply_color=True):
    """
    Apply a series of geometric (physical) augmentations to both image and label:
      - Flipping, rotation, translation, random resized cropping,
        elastic transformation, random scaling, and center cropping.
    Then, if apply_color is True (i.e. for training), apply additional
    noise/color augmentations (Gaussian blur and color jitter) only to the image.
    The label remains unchanged for these color augmentations.
    """
    # Random horizontal flip
    if random.random() < 0.5:
        image = F.hflip(image)
        label = F.hflip(label)
    # Random rotation
    angle = random.uniform(-15, 15)
    image = F.rotate(image, angle, interpolation=Image.BILINEAR)
    label = F.rotate(label, angle, interpolation=Image.NEAREST)
    # Random translation using affine (translation only)
    max_translate = 0.1 * size[0]  # 10% of image size
    translate_x = int(random.uniform(-max_translate, max_translate))
    translate_y = int(random.uniform(-max_translate, max_translate))
    image = F.affine(image, angle=0, translate=(translate_x, translate_y),
                     scale=1.0, shear=0, interpolation=Image.BILINEAR)
    label = F.affine(label, angle=0, translate=(translate_x, translate_y),
                     scale=1.0, shear=0, interpolation=Image.NEAREST)
    # Random resized crop
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
    image = F.resized_crop(image, i, j, h, w, size, interpolation=Image.BILINEAR)
    label = F.resized_crop(label, i, j, h, w, size, interpolation=Image.NEAREST)
    # Elastic transformation
    image, label = elastic_transform_pair(image, label, alpha=34, sigma=4)
    # Additional augmentation: Random scaling (zoom in/out)
    scale_factor = random.uniform(0.9, 1.1)
    new_size = (int(size[0] * scale_factor), int(size[1] * scale_factor))
    image = F.resize(image, new_size, interpolation=Image.BILINEAR)
    label = F.resize(label, new_size, interpolation=Image.NEAREST)
    # Center crop back to desired size
    image = F.center_crop(image, size)
    label = F.center_crop(label, size)
    # For training only, apply color/noise augmentations to the image (labels remain unchanged)
    if apply_color:
        # Gaussian blur (applied with probability 0.3)
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        # Color jitter
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                               saturation=0.2, hue=0.1)
        image = color_jitter(image)
    return image, label

# --------------------------
# OOP Preprocessor Class
# --------------------------

class Preprocessor:
    def __init__(self, raw_color_path, raw_label_path,
                 proc_color_path, proc_label_path,
                 resize_dim=128, do_augmentation=True,
                 is_train=True, max_images=None, aug_count=10):
        """
        Parameters:
          - raw_color_path: Relative path to raw color images.
          - raw_label_path: Relative path to raw label/mask images.
          - proc_color_path: Relative path to save processed color images.
          - proc_label_path: Relative path to save processed label images.
          - resize_dim: Target dimension for resizing/padding.
          - do_augmentation: Whether to augment (only for training).
          - is_train: True if processing training data.
          - max_images: Process only a subset of images (for testing the pipeline).
          - aug_count: Number of augmentations to create per image (train only).
        """
        self.raw_color_path = Path(raw_color_path)
        self.raw_label_path = Path(raw_label_path)
        self.proc_color_path = Path(proc_color_path)
        self.proc_label_path = Path(proc_label_path)
        self.resize_dim = resize_dim
        self.do_augmentation = do_augmentation and is_train
        self.is_train = is_train
        self.max_images = max_images
        self.aug_count = aug_count

        self.proc_color_path.mkdir(parents=True, exist_ok=True)
        self.proc_label_path.mkdir(parents=True, exist_ok=True)

        # Create transforms for images and labels
        self.transform_img = ResizeWithPadding(target_size=224, padding_mode='mean',
                                               force_resize=True, resize_dims=(resize_dim, resize_dim))
        self.transform_label = ResizeWithPaddingLabel(target_size=224, force_resize=True,
                                                      resize_dims=(resize_dim, resize_dim))

    def process(self):
        # Find all color images (assume image file names match label file names)
        image_extensions = (".jpg", ".jpeg", ".png")
        image_files = [f for f in self.raw_color_path.rglob("*") if f.suffix.lower() in image_extensions]
        if self.max_images is not None:
            image_files = image_files[:self.max_images]

        if not image_files:
            print("❌ No images found in", self.raw_color_path)
            return

        for img_file in image_files:
            label_file = self.raw_label_path / f"{img_file.stem}.png"
            if not label_file.exists():
                print(f"⚠️  Label for {img_file.name} not found, skipping.")
                continue

            # Open image and label (assume label is a segmentation mask in grayscale)
            img = Image.open(img_file).convert("RGB")
            label = Image.open(label_file).convert("L")

            # Apply resizing and padding to both image and label
            proc_img = self.transform_img(img)
            proc_label = self.transform_label(label)

            # Save processed (base) image and label
            proc_img.save(self.proc_color_path / f"processed_{img_file.name}")
            proc_label.save(self.proc_label_path / f"processed_{label_file.name}")
            print(f"Processed {img_file.name}")

            # If training and augmentation is enabled, create additional augmented pairs
            if self.is_train and self.do_augmentation:
                for i in range(self.aug_count):
                    aug_img, aug_label = augment_pair(proc_img, proc_label, size=(self.resize_dim, self.resize_dim))
                    aug_img.save(self.proc_color_path / f"processed_{img_file.stem}_aug_{i}{img_file.suffix}")
                    aug_label.save(self.proc_label_path / f"processed_{label_file.stem}_aug_{i}{label_file.suffix}")
                    print(f"Augmented {img_file.name} -> aug {i}")

# --------------------------
# Command Line Interface
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess train/test images with labels/masks using relative paths."
    )
    parser.add_argument("--raw_color", type=str, required=True,
                        help="Relative path to raw color images (e.g., Dataset/raw/TrainVal/color).")
    parser.add_argument("--raw_label", type=str, required=True,
                        help="Relative path to raw label images (e.g., Dataset/raw/TrainVal/label).")
    parser.add_argument("--proc_color", type=str, required=True,
                        help="Relative path to save processed color images (e.g., Dataset/processed/TrainVal/color).")
    parser.add_argument("--proc_label", type=str, required=True,
                        help="Relative path to save processed label images (e.g., Dataset/processed/TrainVal/label).")
    parser.add_argument("--resize_dim", type=int, default=128,
                        help="Output dimension for resizing (e.g., 128 or 256).")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable augmentation (for test sets).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Process only a subset of images (default: all).")
    parser.add_argument("--aug_count", type=int, default=10,
                        help="Number of augmentations per image (train only).")
    parser.add_argument("--set_type", type=str, choices=["TrainVal", "Test"], default="TrainVal",
                        help="Dataset type: TrainVal or Test.")

    args = parser.parse_args()

    # Define relative paths based on set type
    if args.set_type == "TrainVal":
        raw_color = Path("Dataset/raw/TrainVal/color")
        raw_label = Path("Dataset/raw/TrainVal/label")
        proc_color = Path("Dataset/processed/TrainVal/color")
        proc_label = Path("Dataset/processed/TrainVal/label")
    else:  # Test set
        raw_color = Path("Dataset/raw/Test/color")
        raw_label = Path("Dataset/raw/Test/label")
        proc_color = Path("Dataset/processed/Test/color")
        proc_label = Path("Dataset/processed/Test/label")

    preprocessor = Preprocessor(raw_color, raw_label,
                                proc_color, proc_label,
                                resize_dim=args.resize_dim,
                                do_augmentation=not args.no_augment,
                                is_train=(args.set_type == "TrainVal"),
                                max_images=args.max_images,
                                aug_count=args.aug_count)
    preprocessor.process()