"""
Tümör Tespiti - Tek Görüntü Tahmin Scripti
Kullanım: python predict.py --image beyin.jpg
"""

import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# AYARLAR
MODEL_PATH  = "best_model.pth"
NUM_CLASSES = 4
IMG_SIZE    = 224
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]  # klasör sırana göre düzenle
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL YÜKLEMESİ
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1280, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# TAHMİN
def predict(image_path, model):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    print(f"\n📷 Görüntü : {image_path}")
    print(f"🔍 Tahmin  : {CLASSES[pred_idx].upper()} ({probs[pred_idx]*100:.2f}%)")
    print("\n── Tüm olasılıklar ──")
    for i, cls in enumerate(CLASSES):
        bar = "█" * int(probs[i].item() * 30)
        print(f"  {cls:<12} {probs[i]*100:5.2f}%  {bar}")

    return CLASSES[pred_idx], probs[pred_idx].item()


# ANA
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tümör Tespiti Tahmini")
    parser.add_argument("--image", required=True, help="Görüntü dosyası yolu")
    args = parser.parse_args()

    model = load_model()
    predict(args.image, model)