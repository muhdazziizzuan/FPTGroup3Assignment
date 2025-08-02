import os, random, time, json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

DATA_DIR   = Path("data")
BATCH_SIZE = 16
EPOCHS     = 100
LR         = 1e-3              # lower LR for Adam on MobileNet
MODEL_FILE = "pest_classifier2.pth"

class DepthwiseSeparableConv(nn.Module):
    """Depth-wise 3×3 conv → BN → ReLU6  ➔  Point-wise 1×1 conv → BN → ReLU6"""
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True)
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

class MobileNetV1(nn.Module):
    """Minimal MobileNet v1 (α = 1.0) — output logits for `num_classes`."""
    def __init__(self, num_classes: int = 12):
        super().__init__()
        
        c = lambda in_c, out_c, s: DepthwiseSeparableConv(in_c, out_c, s)

        # stem
        layers = [
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # 224→112
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            # (filters, stride) sequence from Table 1
            c(32,  64, 1),      # 112×112
            c(64, 128, 2),      # 56×56
            c(128,128, 1),
            c(128,256, 2),      # 28×28
            c(256,256, 1),
            c(256,512, 2),      # 14×14
        ]

        # 5 × (512, stride 1)
        for _ in range(5):
            layers.append(c(512, 512, 1))

        layers += [
            c(512,1024, 2),     # 7×7
            c(1024,1024,1)      # 7×7
        ]

        self.features = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool2d(1)     # 1×1×1024
        self.classifier = nn.Linear(1024, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

train_tfms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(.1,.1), scale=(.9,1.1)),
    transforms.ColorJitter(.3,.3,.3,.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

print("Loading datasets…")
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
val_ds   = datasets.ImageFolder(DATA_DIR / "test",  transform=val_tfms)
num_classes = len(train_ds.classes)
print(f"✓ {num_classes} classes  →  {train_ds.classes}")

# weighted sampler
label_list     = [lbl for _, lbl in train_ds.samples]
class_counts   = Counter(label_list)
weights_per_cls = torch.tensor([1.0 / class_counts[i] for i in range(num_classes)],
                               dtype=torch.float)
sample_weights = torch.tensor([weights_per_cls[lbl] for lbl in label_list])
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def accuracy(out, y):         # top-1
    return (out.argmax(1) == y).float().mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42);  np.random.seed(42);  random.seed(42)
print(f"✓ Device: {device}")

model = MobileNetV1(num_classes).to(device)
print(f"✓ Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss(weight=weights_per_cls.to(device),
                                label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                 factor=0.5, patience=8,
                                                 min_lr=1e-6)

best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    # ───── train ─────
    model.train()
    tr_loss = tr_corr = 0
    for i,(xb,yb) in enumerate(train_dl, start=1):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tr_loss += loss.item() * xb.size(0)
        tr_corr += (out.argmax(1) == yb).sum().item()

        if i % 50 == 0:
            print(f"\rEpoch {epoch}/{EPOCHS}  batch {i}/{len(train_dl)}  "
                  f"loss {loss.item():.3f}", end="")

    train_acc  = 100 * tr_corr / len(train_ds)
    train_loss = tr_loss / len(train_ds)

    # ───── validate with TTA (orig + h-flip) ─────
    model.eval()
    v_loss = v_corr = 0
    with torch.no_grad():
        for xb,yb in val_dl:
            xb,yb = xb.to(device), yb.to(device)
            out_o = model(xb)
            out_f = model(torch.flip(xb, dims=[3]))   # horizontal flip
            out   = (out_o + out_f) / 2
            v_loss += criterion(out, yb).item() * xb.size(0)
            v_corr += (out.argmax(1) == yb).sum().item()

    val_acc  = 100 * v_corr / len(val_ds)
    val_loss = v_loss / len(val_ds)

    print(f"\nEpoch {epoch:>3}: "
          f"train acc {train_acc:5.1f}%  "
          f"val acc {val_acc:5.1f}%  "
          f"val loss {val_loss:.3f}")

    # save best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "classes": train_ds.classes,
            "class_to_idx": train_ds.class_to_idx
        }, MODEL_FILE)
        print(f"✓ Saved new best model  ({best_acc:.1f} %) →  {MODEL_FILE}")

    scheduler.step(best_acc)

print("\nTraining finished.")
print(f"Best validation accuracy: {best_acc:.1f}%")
