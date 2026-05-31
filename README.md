#  Beyin Tümörü Tespiti — EfficientNet-B0 + Grad-CAM

> Derin öğrenme ile MRI görüntülerinden beyin tümörü sınıflandırması.  
> Transfer Learning, Grad-CAM açıklanabilirliği ve Gradio web arayüzü ile donatılmıştır.

---

##  Sonuçlar

| Metrik | Değer |
|---|---|
| Validation Accuracy | **%95.77** |
| Macro F1-Score | **%96** |
| En İyi Epoch | 16 |
| Epoch Süresi | ~4.4 saniye |
| GPU | RTXP 6000 |

### Sınıf Bazında Performans

| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | %100 | %86 | %92 |
| Meningioma | %92 | %98 | %95 |
| No Tumor | %95 | %100 | %97 |
| Pituitary | %98 | %99 | %99 |

---

##  Proje Yapısı

```
beyin-tumoru-tespiti/
│
├── tumor_dataset/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── val/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── tumor_train.py        # Eğitim scripti
├── predict.py            # Tek görüntü tahmin
├── app.py                # Gradio arayüzü + Grad-CAM
├── best_model.pth        # En iyi model ağırlıkları
├── confusion_matrix.png  # Confusion matrix görseli
├── training_curves.png   # Eğitim eğrileri
└── README.md
```

---

##  Kurulum

```bash
pip install torch torchvision gradio opencv-python matplotlib seaborn scikit-learn
```

---

##  Kullanım

### 1. Eğitim

```bash
python tumor_train.py
```

Eğitim tamamlandığında `best_model.pth` ve `training_curves.png` otomatik kaydedilir.

### 2. Tek Görüntü Tahmini

```bash
python predict.py --image beyin_mri.jpg
```

Çıktı örneği:
```
 Görüntü  : beyin_mri.jpg
 Tahmin   : GLIOMA (94.32%)

── Tüm olasılıklar ──
  glioma       94.32%  ██████████████████████████████
  meningioma    3.12%  █
  notumor       1.44%
  pituitary     1.12%
```

### 3. Gradio Arayüzü (Grad-CAM dahil)

```bash
python app.py
```

Tarayıcıda açılır, MRI yükle → tahmin + ısı haritası yan yana görürsün.

---

##  Model Mimarisi

```
EfficientNet-B0 (ImageNet pretrained)
        ↓
   features (dondurulmadı, fine-tune edildi)
        ↓
   Dropout(p=0.4)
        ↓
   Linear(1280 → 4)
        ↓
   Softmax → [glioma, meningioma, notumor, pituitary]
```

### Eğitim Parametreleri

```python
BATCH_SIZE  = 64
NUM_EPOCHS  = 20
LR          = 5e-4
WEIGHT_DECAY= 1e-3
OPTIMIZER   = Adam
SCHEDULER   = CosineAnnealingLR
PRECISION   = Mixed (FP16)
```

---

##  Grad-CAM

Modelin MRI görüntüsünün **hangi bölgesine bakarak** karar verdiğini görselleştirir.  
EfficientNet-B0'ın son konvolüsyon katmanı (`model.features[-1][0]`) hedef alınmıştır.

---

##  Veri Seti

**Brain Tumor MRI Dataset** — Masoud Nickparvar  
[Kaggle'da görüntüle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Sınıf | Görüntü Sayısı |
|---|---|
| Glioma | ~1800 |
| Meningioma | ~1700 |
| No Tumor | ~1600 |
| Pituitary | ~1800 |
| **Toplam** | **~7000** |

---

##  Geliştiriciler

**Mehmet Kalbişen** — 032390011  
**Emrecan Kutlu** — 032390027   
**Efe Tutucu** — 032390034