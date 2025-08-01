import os, torch, time
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import random
import numpy as np
from collections import Counter

# --- Constants ---------------------------------------------
DATA_DIR = Path("data")
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.001
MODEL_FILE = "pest_classifier2.pth"

# --- Model Architecture ------------------------------------
class ImprovedPestCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(ImprovedPestCNN, self).__init__()
        
        # Deeper feature extraction
        self.features = nn.Sequential(
            # Block 1: Edge detection
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: Pattern detection
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: Complex features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: High-level features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # MODIFICATION: Use Kaiming for Linear layers as well
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# --- Data Augmentation -------------------------------------
train_tfms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load and Analyze Datasets ------------------------------
print("Loading datasets...")
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
val_ds = datasets.ImageFolder(DATA_DIR / "test", transform=val_tfms)

num_classes = len(train_ds.classes)
print(f"✓ Found {num_classes} classes: {train_ds.classes}")

train_labels = [label for _, label in train_ds.samples]
class_counts = sorted(Counter(train_labels).items())
class_weights_values = [count for _, count in class_counts]

# Create weighted sampler for imbalanced classes
class_weights = 1.0 / torch.tensor(class_weights_values, dtype=torch.float)
sample_weights = [class_weights[label] for _, label in train_ds.samples]
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Create data loaders with sampler
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Main Execution Block -----------------------------------
if __name__ == '__main__':
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    model = ImprovedPestCNN(num_classes=num_classes).to(device)
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # MODIFICATION: Add Label Smoothing to the loss function
    class_weights_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    # MODIFICATION: Use a more stable Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-7)
    
    # MODIFICATION: Use a more patient learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
    )
    
    print(f"\nStarting training...")
    print("="*60)
    
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar_desc = f"Epoch {epoch+1}/{EPOCHS}"
        
        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i > 0 and i % 50 == 0:
                print(f'\r{pbar_desc} - Batch {i}/{len(train_dl)} - Loss: {loss.item():.3f} - Acc: {100.*correct/total:.1f}%', end='')
        
        train_loss = running_loss / len(train_dl)
        train_acc = 100. * correct / total
        
        # Validation phase with Test-Time Augmentation (TTA)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Original image predictions
                outputs_orig = model(inputs)
                
                # Horizontally flipped image predictions
                flipped_inputs = transforms.functional.hflip(inputs)
                outputs_flipped = model(flipped_inputs)
                
                # MODIFICATION: Average the predictions from original and flipped
                outputs = (outputs_orig + outputs_flipped) / 2.0
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_dl)
        val_acc = 100. * correct / total
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'\n{pbar_desc} - Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Val Loss: {val_loss:.3f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'classes': train_ds.classes,
                'class_to_idx': train_ds.class_to_idx,
                'history': history
            }, MODEL_FILE)
            print(f'✓ New best model saved: {best_acc:.1f}%')
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
    print(f"\n{'='*60}")
    print(f"✓ Training complete! Best accuracy: {best_acc:.1f}% saved to {MODEL_FILE}")