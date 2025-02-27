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

def augment_pair(image, label, size=(256, 256), apply_color=True, **params):
    """
    Applies a random subset of augmentations in random order.
    Each augmentation is stored as a function with its own probability.
    
    To disable an augmentation, set its probability to 0.
    To guarantee an augmentation, set its probability to 1.
    Modify parameters (e.g., rotation_angle, translate_factor) to adjust intensity.
    """
    # Build list of augmentation functions with their associated probability.
    augmentations = []

    # Horizontal flip
    augmentations.append((
        lambda img, lbl: (F.hflip(img), F.hflip(lbl)),
        params.get("flip_prob", 0.5)  # default: 50% chance
    ))

    # Rotation (you can control intensity via "rotation_angle")
    def rotate_aug(img, lbl):
        angle = random.uniform(-params.get("rotation_angle", 5), params.get("rotation_angle", 10))
        return F.rotate(img, angle, interpolation=Image.BILINEAR), F.rotate(lbl, angle, interpolation=Image.NEAREST)
    augmentations.append((
        rotate_aug,
        params.get("rotation_prob", 0.25)  # default: always apply rotation
    ))

    # Translation (intensity via "translate_factor")
    def translate_aug(img, lbl):
        max_translate = params.get("translate_factor", 0.05) * size[0]
        tx = int(random.uniform(-max_translate, max_translate))
        ty = int(random.uniform(-max_translate, max_translate))
        return (F.affine(img, angle=0, translate=(tx, ty), scale=1.0, shear=0, interpolation=Image.BILINEAR),
                F.affine(lbl, angle=0, translate=(tx, ty), scale=1.0, shear=0, interpolation=Image.NEAREST))
    augmentations.append((
        translate_aug,
        params.get("translate_prob", 0.05)  # default: always apply translation
    ))

    # Random Resized Crop (controls framing; intensity via crop_scale_range and crop_ratio_range)
    def crop_aug(img, lbl):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img,
            scale=params.get("crop_scale_range", (0.9, 1.0)),
            ratio=params.get("crop_ratio_range", (1.0, 1.0))
        )
        return (F.resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR),
                F.resized_crop(lbl, i, j, h, w, size, interpolation=Image.NEAREST))
    augmentations.append((
        crop_aug,
        params.get("crop_prob", 0.01)  # default: always apply crop
    ))

    # Elastic Transformation (intensity via "elastic_alpha" and "elastic_sigma")
    def elastic_aug(img, lbl):
        return elastic_transform_pair(
            img, lbl,
            alpha=params.get("elastic_alpha", 15),
            sigma=params.get("elastic_sigma", 2)
        )
    augmentations.append((
        elastic_aug,
        params.get("elastic_prob", 0.01)  # default: always apply elastic transformation
    ))

    # Random Scaling (intensity via "scaling_range")
    def scaling_aug(img, lbl):
        scale_factor = random.uniform(*params.get("scaling_range", (0.5, 1.5)))
        new_size = (int(size[0] * scale_factor), int(size[1] * scale_factor))
        return (F.resize(img, new_size, interpolation=Image.BILINEAR),
                F.resize(lbl, new_size, interpolation=Image.NEAREST))
    augmentations.append((
        scaling_aug,
        params.get("scaling_prob", 0.0)  # default: always apply scaling
    ))

    # Color augmentations: Gaussian blur and Color jitter.
    def color_aug(img, lbl):
        if random.random() < params.get("blur_prob", 0.0):
            img = img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(*params.get("blur_radius_range", (0.5, 1.5)))
                )
            )
        color_jitter = transforms.ColorJitter(**params.get("color_jitter_params", {
            'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1
        }))
        img = color_jitter(img)
        return img, lbl
    augmentations.append((
        color_aug,
        params.get("color_prob", 0.25)  # default: always apply color adjustments if apply_color is True
    ))

    # Optionally randomize the order of augmentations.
    random.shuffle(augmentations)

    # Apply each augmentation based on its probability.
    for func, prob in augmentations:
        if random.random() < prob:
            image, label = func(image, label)

    # Optionally, enforce a final center crop to ensure the image is the desired size.
    image = F.center_crop(image, size)
    label = F.center_crop(label, size)

    return image, label

# --------------------------
# OOP Preprocessor Class
# --------------------------

class Preprocessor:
    def __init__(self, raw_color_path, raw_label_path,
                 proc_color_path, proc_label_path,
                 resize_dim=128, do_augmentation=True,
                 is_train=True, max_images=None, aug_count=10, aug_params=None):
        
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
        self.aug_params = aug_params if aug_params is not None else {}

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
                    aug_img, aug_label = augment_pair(proc_img, proc_label, size=(self.resize_dim, self.resize_dim), **self.aug_params)
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
    parser.add_argument("--flip_prob", type=float, default=0.5, help="Probability for horizontal flip.")
    parser.add_argument("--rotation_angle", type=float, default=5, help="Maximum rotation angle (in degrees).")
    parser.add_argument("--translate_factor", type=float, default=0.05, help="Translation factor (fraction of image size).")
    parser.add_argument("--crop_scale_min", type=float, default=0.9, help="Minimum scale for random resized crop.")
    parser.add_argument("--crop_scale_max", type=float, default=1.0, help="Maximum scale for random resized crop.")
    parser.add_argument("--crop_ratio_min", type=float, default=1.0, help="Minimum aspect ratio for random resized crop.")
    parser.add_argument("--crop_ratio_max", type=float, default=1.0, help="Maximum aspect ratio for random resized crop.")
    parser.add_argument("--elastic_alpha", type=float, default=15, help="Elastic transformation alpha parameter.")
    parser.add_argument("--elastic_sigma", type=float, default=2, help="Elastic transformation sigma parameter.")
    parser.add_argument("--scaling_min", type=float, default=0.0, help="Minimum scaling factor for random scaling.")
    parser.add_argument("--scaling_max", type=float, default=0.0, help="Maximum scaling factor for random scaling.")
    parser.add_argument("--blur_prob", type=float, default=0.3, help="Probability of applying Gaussian blur.")
    parser.add_argument("--blur_radius_min", type=float, default=0.5, help="Minimum radius for Gaussian blur.")
    parser.add_argument("--blur_radius_max", type=float, default=1.5, help="Maximum radius for Gaussian blur.")
    parser.add_argument("--color_jitter_brightness", type=float, default=0.2, help="Brightness for color jitter.")
    parser.add_argument("--color_jitter_contrast", type=float, default=0.2, help="Contrast for color jitter.")
    parser.add_argument("--color_jitter_saturation", type=float, default=0.2, help="Saturation for color jitter.")
    parser.add_argument("--color_jitter_hue", type=float, default=0.1, help="Hue for color jitter.")
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
    
    aug_params = {
        "flip_prob": args.flip_prob,  # Already exists.
        "rotation_angle": args.rotation_angle,
        "rotation_prob": 0.25,         # Always apply rotation, for instance.
        "translate_factor": args.translate_factor,
        "translate_prob": 0.05,        # Always apply translation.
        "crop_scale_range": (args.crop_scale_min, args.crop_scale_max),
        "crop_ratio_range": (args.crop_ratio_min, args.crop_ratio_max),
        "crop_prob": 0.01,             # Always apply crop.
        "elastic_alpha": args.elastic_alpha,
        "elastic_sigma": args.elastic_sigma,
        "elastic_prob": 0.01,          # Always apply elastic transform.
        "scaling_range": (args.scaling_min, args.scaling_max),
        "scaling_prob": 0.0,          # Always apply scaling.
        "blur_prob": 0.0,
        "blur_radius_range": (args.blur_radius_min, args.blur_radius_max),
        "color_jitter_params": {
            "brightness": args.color_jitter_brightness,
            "contrast": args.color_jitter_contrast,
            "saturation": args.color_jitter_saturation,
            "hue": args.color_jitter_hue,
        },
        "color_prob": 1.0            # Always apply color adjustments if enabled.
    }

    preprocessor = Preprocessor(raw_color, raw_label,
                                proc_color, proc_label,
                                resize_dim=args.resize_dim,
                                do_augmentation=not args.no_augment,
                                is_train=(args.set_type == "TrainVal"),
                                max_images=args.max_images,
                                aug_count=args.aug_count,
                                aug_params=aug_params)

    preprocessor.process()


    '''
    python CV-CW1/src/preprocessing.py --raw_color Dataset/raw/Test/color \
                        --raw_label Dataset/raw/Test/label \
                        --proc_color Dataset/processed/Test/color \
                        --proc_label Dataset/processed/Test/label \
                        --resize_dim 256 \
                        --no_augment \
                        --set_type Test'''

'''
    python CV-CW1/src/preprocessing.py --raw_color Dataset/raw/TrainVal/color \
                        --raw_label Dataset/raw/TrainVal/label \
                        --proc_color Dataset/processed/TrainVal/color \
                        --proc_label Dataset/processed/TrainVal/label \
                        --resize_dim 256 \
                        --aug_count 10 \
                        --set_type TrainVal \
                        --translate_factor 0.02 \
                        --elastic_alpha 0.0 \
                        --elastic_sigma 0.05 \
                        --blur_radius_min 0.0 \
                        --blur_radius_max 0.5
                        '''