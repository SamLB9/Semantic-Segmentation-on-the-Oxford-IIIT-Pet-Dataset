import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset  # Added Subset here
import os
import sys
import csv
from tqdm import tqdm  # Add this import at the top
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
from segmentation_dataset import SegmentationDataset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

# Add the CV-CW1 directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
cw1_dir = os.path.dirname(script_dir)
sys.path.insert(0, cw1_dir)

from debug_utils import compute_iou, get_weighted_ce_loss, adjust_learning_rate, visualize_samples, visualize_predictions, compute_class_weights
from models.unet import UNet  # Use relative import if unet.py is in the same directory

# Define Dice loss
def dice_loss(pred, target, smooth=1e-6):
    num_classes = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float().contiguous()
    intersection = (pred * target_one_hot).sum(dim=(2,3))
    pred_sum = pred.sum(dim=(2,3))
    target_sum = target_one_hot.sum(dim=(2,3))
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return 1 - dice.mean()

# Hybrid loss combining weighted cross-entropy and Dice loss (50/50)
class HybridLoss(nn.Module):
    def __init__(self, weight=0.5, ce_weight=None):
        super().__init__()
        self.weight = weight
        # Use weighted cross-entropy if ce_weight is provided
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight) if ce_weight is not None else nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        d  = dice_loss(pred, target)
        return self.weight * ce + (1 - self.weight) * d

# We'll update the criterion later once we compute class weights.
criterion = None

def pixel_accuracy(output, target):
    _, preds = torch.max(output, dim=1)
    correct = (preds == target).float().sum()
    return correct / torch.numel(target)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += pixel_accuracy(outputs, masks).item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        # Print distribution info for first two batches of each epoch
        if batch_idx < 2:
            unique_pred, counts_pred = preds.unique(return_counts=True)
            unique_mask, counts_mask = masks.unique(return_counts=True)
            ##print(f"Train batch {batch_idx} - preds dist:", dict(zip(unique_pred.tolist(), counts_pred.tolist())))
            ##print(f"Train batch {batch_idx} - masks dist:", dict(zip(unique_mask.tolist(), counts_mask.tolist())))
        # Update progress bar description with the current loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(dataloader.dataset), running_acc / len(dataloader.dataset)

def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            running_acc += pixel_accuracy(outputs, masks).item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            # Print distribution info for first two batches
            if batch_idx < 2:
                unique_pred, counts_pred = preds.unique(return_counts=True)
                unique_mask, counts_mask = masks.unique(return_counts=True)
                ##print(f"Val batch {batch_idx} - preds dist:", dict(zip(unique_pred.tolist(), counts_pred.tolist())))
                ##print(f"Val batch {batch_idx} - masks dist:", dict(zip(unique_mask.tolist(), counts_mask.tolist())))
    return running_loss / len(dataloader.dataset), running_acc / len(dataloader.dataset)

def evaluate_iou(model, dataloader, device, num_classes=4):
    model.eval()
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.append(preds)
            all_masks.append(masks.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    return compute_iou(all_preds, all_masks, num_classes)

def create_weighted_dataloader(dataset, batch_size=8):
    # Compute pixel counts for each class across the whole dataset
    class_weights = compute_class_weights(dataset, num_classes=4)
    # For each sample in the dataset, compute an approximate sample weight (e.g. average weight in its mask)
    sample_weights = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        # Compute the weight as a weighted average of pixel weights (convert to float tensor)
        mask_float = mask.float()
        weights = torch.zeros_like(mask_float)
        for cls in range(4):
            weights[mask == cls] = class_weights[cls]
        sample_weights.append(weights.mean().item())
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Train the model on the entire dataset (using train/val split) for 100 epochs.
def train_model(dataset, num_epochs=100, batch_size=8):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Create a train/validation split
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    # Option 1: Regular loader
    # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # Option 2: Oversampled loader
    train_loader = create_weighted_dataloader(train_subset, batch_size)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    
    # Compute class weights from the training set and update criterion accordingly
    class_weights = compute_class_weights(train_subset, num_classes=4)
    print("Class weights:", class_weights)
    global criterion
    criterion = HybridLoss(weight=0.5, ce_weight=class_weights).to(device)
    
    model = UNet(n_channels=3, n_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    metrics = []
    
    # Initialize EMA variables and smoothing factor (alpha)
    ema_alpha = 0.3
    ema_val_loss = None
    ema_val_acc = None
    ema_val_iou = None
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, device)
        val_iou = evaluate_iou(model, val_loader, device, 4)

        # Compute exponential moving averages
        if ema_val_loss is None:
            ema_val_loss = val_loss
            ema_val_acc = val_acc
            ema_val_iou = val_iou
        else:
            ema_val_loss = ema_alpha * val_loss + (1 - ema_alpha) * ema_val_loss
            ema_val_acc = ema_alpha * val_acc + (1 - ema_alpha) * ema_val_acc
            ema_val_iou = ema_alpha * val_iou + (1 - ema_alpha) * ema_val_iou

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f} (EMA: {ema_val_loss:.4f}), "
              f"Val Acc: {val_acc:.4f} (EMA: {ema_val_acc:.4f}), "
              f"Val IoU: {val_iou:.4f} (EMA: {ema_val_iou:.4f})")
        
        # <--- Debug: display one sample prediction from the val set
        if epoch % 5 == 0:
            sample_images, sample_masks = next(iter(val_loader))
            sample_images = sample_images.to(device)
            outputs = model(sample_images)
            preds = torch.argmax(outputs, dim=1).cpu()
            print("Sample prediction unique values:", preds[0].unique())
        
        scheduler.step(val_loss)
        
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = "unet_model_best.pth"
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_model_path, metrics

def test_model(model_path, test_dataset, batch_size=8):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    running_loss, running_acc = 0.0, 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            # Get raw logits then apply softmax
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            running_acc += pixel_accuracy(outputs, masks).item() * images.size(0)
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_acc / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return test_loss, test_acc

def save_metrics(metrics, test_loss, test_acc, filepath):
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)
        
        writer.writerow({
            'epoch': 'test',
            'train_loss': '',
            'train_acc': '',
            'val_loss': test_loss,
            'val_acc': test_acc
        })

# Add a remap function to convert mask intensities to class indices
def remap_mask(mask):
    # Convert PIL Image to NumPy array
    mask_np = np.array(mask)
    # Define mapping from original intensities to new class indices
    mapping = {0: 0, 38: 1, 75: 2, 255: 3}
    remapped = np.zeros_like(mask_np)
    for orig, new in mapping.items():
        remapped[mask_np == orig] = new
    # Convert to torch tensor of type long
    return torch.from_numpy(remapped).long()

def main():
    print("Starting main()")
    sys.stdout.flush()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    sys.stdout.flush()
    
    transform_img = transforms.Compose([transforms.ToTensor()])
    transform_label = remap_mask

    dataset = SegmentationDataset(root_dir="Dataset/processed/TrainVal",
                                  transform_img=transform_img,
                                  transform_label=transform_label)
    
    # visualize_samples(dataset)
    
    print("Dataset root:", "Dataset/processed/TrainVal")
    print("Image directory:", dataset.img_dir)
    print(f"Found {len(dataset)} images in the dataset.")
    sys.stdout.flush()
    if len(dataset) == 0:
        print("No images found. Check your directory structure and paths.")
        sys.stdout.flush()
        return

    # Use a small number of epochs for testing (e.g., 3)
    best_model_path, metrics = train_model(dataset, num_epochs=50, batch_size=8)

    # Visualize predictions on a few validation samples
    print("Visualizing predictions on validation samples:")
    # You can pass the validation dataloader to visualize_predictions. Here we reuse a subset:
    val_loader = DataLoader(Subset(dataset, list(range(10))), batch_size=2)
    visualize_predictions(model=UNet(n_channels=3, n_classes=4).to(device),
                          dataloader=val_loader,
                          device=device,
                          num_samples=3)

    test_dataset = SegmentationDataset(root_dir="Dataset/processed/Test",
                                       transform_img=transform_img,
                                       transform_label=transform_label)
    test_loss, test_acc = test_model(best_model_path, test_dataset, batch_size=8)

    save_metrics(metrics, test_loss, test_acc, 'results/training_metrics.csv')

    print("Done")
    sys.stdout.flush()
    time.sleep(5)

if __name__ == "__main__":
    main()
