"""
Kullanım: python app.py
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import gradio as gr
from torchvision import transforms, models
from PIL import Image

# AYARLAR
MODEL_PATH  = "best_model.pth"
NUM_CLASSES = 4
IMG_SIZE    = 224
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SINIF_ACIKLAMA = {
    "glioma"     : "Glioma — beyin veya omuriliğin glial hücrelerinden kaynaklanan tümör.",
    "meningioma" : "Meningioma — beyin zarlarından (meninges) kaynaklanan genellikle iyi huylu tümör.",
    "notumor"    : "Tümör tespit edilmedi — görüntüde anormal bir kitle bulunmuyor.",
    "pituitary"  : "Pituitary — hipofiz bezinde oluşan tümör.",
}

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
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

model = load_model()
print(f"✓ Model yüklendi — {DEVICE}")

# ─────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model      = model
        self.gradients  = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# EfficientNet-B0'ın son conv katmanı
target_layer = model.features[-1][0]
gradcam      = GradCAM(model, target_layer)


def apply_heatmap(original_pil, cam):
    """Grad-CAM ısı haritasını orijinal görüntünün üstüne bindirme"""
    orig = np.array(original_pil.resize((IMG_SIZE, IMG_SIZE)))
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * orig + 0.5 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


# ANA TAHMİN FONKSİYONU
def tahmin_et(pil_image):
    if pil_image is None:
        return None, {}, ""

    img_rgb = pil_image.convert("RGB")
    tensor  = transform(img_rgb).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    # Tahmin
    output   = model(tensor)
    probs    = torch.softmax(output, dim=1)[0]
    pred_idx = probs.argmax().item()
    pred_cls = CLASSES[pred_idx]

    # Grad-CAM
    cam     = gradcam.generate(tensor, pred_idx)
    overlay = apply_heatmap(img_rgb, cam)

    # Olasılık dict 
    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    aciklama = f"**{pred_cls.upper()}** — {SINIF_ACIKLAMA[pred_cls]}"

    return overlay, prob_dict, aciklama


# GRADIO ARAYÜZÜ
with gr.Blocks(title="Beyin Tümörü Tespiti", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧠 Beyin Tümörü Tespiti
    MRI görüntüsü yükleyin — model tahmin yapar ve **Grad-CAM** ile odak bölgesini gösterir.
    """)

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="MRI Görüntüsü Yükle")
            btn       = gr.Button("Analiz Et", variant="primary")

        with gr.Column():
            img_output = gr.Image(label="Grad-CAM Isı Haritası")
            label_out  = gr.Label(label="Sınıf Olasılıkları", num_top_classes=4)
            text_out   = gr.Markdown()

    btn.click(
        fn=tahmin_et,
        inputs=img_input,
        outputs=[img_output, label_out, text_out]
    )

    gr.Examples(
        examples=[],  
        inputs=img_input
    )

if __name__ == "__main__":
    demo.launch(share=True)  