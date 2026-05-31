import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224

# Model yükle
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1280, 4))
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.to(DEVICE).eval()

# Val seti
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_loader = DataLoader(
    datasets.ImageFolder("./dataset/val", transform=transform),
    batch_size=64, shuffle=False
)

# Tahminler
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(DEVICE))
        all_preds  += outputs.argmax(1).cpu().tolist()
        all_labels += labels.tolist()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Kaydedildi: confusion_matrix.png")

# Classification Report
print("\n── Classification Report ──")
print(classification_report(all_labels, all_preds, target_names=CLASSES))