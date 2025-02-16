import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

class ResizeWithPadding:
    def __init__(self, target_size=224, padding_mode='hybrid', force_resize=False, resize_dims=(128, 128)):
        """
        Resize and pad an image while maintaining its aspect ratio.

        Parameters:
        - target_size (int): The desired size of the output square image (e.g., 224, 256).
        - padding_mode (str): Padding type ('mean' for mean pixel padding, 'reflect' for reflection padding, 'hybrid' for both).
        - force_resize (bool): If True, resizes shorter side to `resize_dims[0]` while maintaining aspect ratio, then pads to `resize_dims`.
        - resize_dims (tuple): The final dimensions to resize/pad to when `force_resize` is enabled.
        """
        self.target_size = target_size
        self.force_resize = force_resize
        self.resize_dims = resize_dims
        assert padding_mode in ['mean', 'reflect', 'hybrid'], "Padding mode must be 'mean', 'reflect', or 'hybrid'"
        self.padding_mode = padding_mode
    
    def resize_image(self, image):
        """Resize the shorter side of the image to match resize_dims[0] while maintaining aspect ratio."""
        if self.force_resize:
            h, w = image.shape[:2]
            scale = self.resize_dims[0] / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized_image
        
        h, w = image.shape[:2]
        scale = self.target_size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def pad_image(self, image):
        """Pad the image to make it square using the selected padding method."""
        h, w = image.shape[:2]
        target_h, target_w = self.resize_dims if self.force_resize else (self.target_size, self.target_size)
        delta_w = target_w - w
        delta_h = target_h - h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        if self.padding_mode == 'mean':
            mean_pixel = np.mean(image, axis=(0, 1), dtype=int)  # Compute mean pixel per channel
            padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=mean_pixel.tolist())
        elif self.padding_mode == 'reflect':
            padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
        else:  # 'hybrid'
            # Apply reflection padding first
            reflected_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
            
            # Compute mean pixel and apply mean padding only if necessary
            mean_pixel = np.mean(image, axis=(0, 1), dtype=int)
            padded_image = cv2.copyMakeBorder(reflected_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=mean_pixel.tolist())
        
        return padded_image
    
    def __call__(self, img):
        """Apply resizing and padding transformation to an image."""
        img = np.array(img)  # Convert PIL image to NumPy array
        resized_img = self.resize_image(img)
        padded_img = self.pad_image(resized_img)
        return Image.fromarray(padded_img)  # Convert back to PIL image

# Example usage
if __name__ == "__main__":
    transform = ResizeWithPadding(target_size=224, padding_mode='hybrid', force_resize=True, resize_dims=(128, 128))  # Choose settings
    img = Image.open("example.jpg").convert("RGB")  # Load an example image
    transformed_img = transform(img)
    transformed_img.show()  # Display the transformed image
