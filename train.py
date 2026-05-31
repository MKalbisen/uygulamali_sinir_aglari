"""
CNN Tümör Tespiti - PyTorch Eğitim Scripti
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# 1. AYARLAR
DATA_DIR    = "/teamspace/studios/this_studio/tumor_dataset"        
NUM_CLASSES = 4                  # glioma, meningioma, pituitary, no_tumor
BATCH_SIZE  = 64                
NUM_EPOCHS  = 20
LR          = 5e-4
IMG_SIZE    = 224
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Cihaz: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. TRANSFORM & DATASET
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Sınıflar: {train_dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# 3. MODEL (Transfer Learning - EfficientNet-B0)
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Son katmanı sınıf sayısına göre güncelle (Dropout 0.4 ile overfit azaltılır)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(1280, NUM_CLASSES)
)

model = model.to(DEVICE)

# 4. LOSS, OPTIMIZER, SCHEDULER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler    = GradScaler()  

# 5. EĞİTİM DÖNGÜSÜ
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_acc   = 0.0
best_model = copy.deepcopy(model.state_dict())

def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

    return total_loss / total, correct / total


print("\n── Eğitim Başlıyor ──")
for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   training=False)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - t0
    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"({elapsed:.1f}s)")

    if val_acc > best_acc:
        best_acc   = val_acc
        best_model = copy.deepcopy(model.state_dict())
        torch.save(best_model, "best_model.pth")
        print(f"  ✓ En iyi model kaydedildi (val_acc={best_acc:.4f})")

print(f"\nEğitim tamamlandı. En iyi Val Acc: {best_acc:.4f}")

# 6. GRAFİK
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"],   label="Val")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"],   label="Val")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Grafik kaydedildi: training_curves.png")