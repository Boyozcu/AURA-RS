from google.colab import drive
drive.mount('/content/drive')

# 1. Önce olası eski sürümleri kaldıralım (isteğe bağlı ama emin olmak için iyi bir adım)
!pip uninstall -y transformers peft

# 2. Kütüphaneleri kuralım
!pip install -q \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    timm \
    transformers>=4.36.0 \
    datasets>=2.12.0 \
    scikit-learn>=1.2.0 \
    peft>=0.10.0 \
    fpdf2>=2.7.0 \
    tqdm>=4.65.0 \
    torchgeo \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    Pillow>=9.5.0

# 3. Kurulum sonrası sürümleri kontrol edelim
import importlib
try:
    import transformers
    print(f"Kurulan Transformers Sürümü: {transformers.__version__}")
except ImportError:
    print("Transformers kütüphanesi kurulamadı veya bulunamadı.")

try:
    import peft
    print(f"Kurulan PEFT Sürümü: {peft.__version__}")
except ImportError:
    print("PEFT kütüphanesi kurulamadı veya bulunamadı.")

print("\n--- ÖNEMLİ ---")
print("LÜTFEN ŞİMDİ COLAB ÇALIŞMA ZAMANINI (RUNTIME) YENİDEN BAŞLATIN!")
print("Menü: 'Çalışma zamanı' > 'Oturumu yeniden başlat' (veya 'Runtime' > 'Restart session')")
print("Yeniden başlattıktan sonra import kodunuzu içeren hücreyi çalıştırın.")

import torch
import torchvision
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor, EncoderDecoderCache # Test için EncoderDecoderCache'i de ekledim
from peft import get_peft_model, LoraConfig
from PIL import Image
from tqdm import tqdm
import timm
import sklearn
import fpdf
import torchgeo
import numpy
import pandas
import matplotlib

print("Tüm kütüphaneler başarıyla import edildi!")

# İsteğe bağlı: Sürümleri tekrar kontrol edin
import transformers
import peft
print(f"Kullanılan Transformers Sürümü: {transformers.__version__}")
print(f"Kullanılan PEFT Sürümü: {peft.__version__}")

