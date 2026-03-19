# ============================================================================
# CELL 01: MOUNT GOOGLE DRIVE & ESTABLISH A PATH
# ============================================================================
# Run this cell FIRST. All data will be saved to Google Drive
# so it won't be lost when Colab disconnects.
# ============================================================================

import os

# ── 1. Mount Google Drive ──
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ── 2. Create a working folder on Google Drive ──
BASE_DIR = '/content/drive/MyDrive/DOCTOR_PHD/FINAL PROJECT/04_RESULT_TRAIN_KARTHY/YOLO_Denoise_Experiment_Karthy'
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)

print(f"✅ Working directory: {os.getcwd()}")
print(f"✅ All datasets, models, and results will be stored here.")


# ============================================================================
# CELL 02: INSTALL THE LIBRARY (Google Colab)
# ============================================================================
import subprocess, sys

def pip_install(*packages):
    for pkg in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

# PyTorch (Colab is available, local installation is required.)
pip_install('numpy')
pip_install('torch', 'torchvision')
pip_install('Pillow')

# Main libraries
pip_install('ultralytics>=8.3.0', 'pandas', 'tabulate', 'matplotlib', 'seaborn', 'pyyaml')
pip_install('bm3d')
pip_install('kagglehub==1.0.0')
pip_install('opencv-python')

print("✅ Library installation complete!")


# ============================================================================
# CELL 03: LIBRARY IMPORT
# ============================================================================
import os
import re
import sys
import gc
import yaml
import shutil
import glob
import time
import warnings
import platform
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from ultralytics import YOLO
import bm3d

warnings.filterwarnings('ignore')

# ============================================================
# ROOT PATH - Set in Cell 0 (Google Drive)
# ============================================================
BASE_DIR = '/content/drive/MyDrive/DOCTOR_PHD/FINAL PROJECT/04_RESULT_TRAIN_KARTHY/YOLO_Denoise_Experiment_Karthy'
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)

print(f"✅ Working directory: {os.getcwd()}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    GPU_MEM_GB = props.total_memory / 1024**3
    print(f"✅ GPU Memory: {GPU_MEM_GB:.1f} GB")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🚀 A100 GPU Optimizations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    torch.backends.cudnn.benchmark = True           # Auto-tune convolution kernels
    torch.backends.cuda.matmul.allow_tf32 = True    # TF32 on Tensor Cores (19x faster matmul)
    torch.backends.cudnn.allow_tf32 = True          # TF32 for cuDNN convolutions
    torch.set_float32_matmul_precision('high')      # Enable TF32 globally
    print("✅ A100 Optimizations: cuDNN benchmark + TF32 Tensor Cores enabled")
else:
    GPU_MEM_GB = 0
    print("⚠️  Without a GPU, it will run on the CPU (which is much slower).")


# ============================================================================
# CELL 04: DOWNLOAD DATASET
# ============================================================================
import kagglehub

DEST_DATASET = os.path.join(BASE_DIR, 'pedestrian-detection')

if os.path.exists(DEST_DATASET) and len(os.listdir(DEST_DATASET)) > 0:
    print(f"✅ The dataset is available at: {DEST_DATASET}")
    print("   (Delete this folder if you want to download again)")
else:
    print("⏬ Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("karthika95/pedestrian-detection")
    print(f"   Downloaded to: {path}")

    # Copy it to the project folder.
    if os.path.exists(DEST_DATASET):
        shutil.rmtree(DEST_DATASET)
    print(f"   Copying to: {DEST_DATASET}")
    shutil.copytree(path, DEST_DATASET)
    print("✅ Dataset downloaded & copied!")

# Display structure
print(f"\n📁 Data structure:")
for root, dirs, files in os.walk(DEST_DATASET):
    level = root.replace(DEST_DATASET, '').count(os.sep)
    if level > 3:
        continue
    indent = '  ' * level
    n = len(files)
    print(f"{indent}📁 {os.path.basename(root)}/ ({n} files)")


# ============================================================================
# CELL 05: PASCAL VOC CONVERSION → YOLO FORMAT
# ============================================================================
SOURCE_DIR = os.path.join(BASE_DIR, 'pedestrian-detection')
YOLO_DIR = os.path.join(BASE_DIR, 'dataset_yolo')

# ⚡ Skip if you have already converted
_data_yaml_check = os.path.join(YOLO_DIR, "data.yaml")
_skip_conversion = False
if os.path.exists(_data_yaml_check):
    _n = len(glob.glob(os.path.join(YOLO_DIR, 'images', 'train', '*')))
    if _n > 0:
        _skip_conversion = True
        DATA_YAML_PATH = _data_yaml_check
        print(f"⚡ The YOLO dataset already exists ({_n} train images) → skip")
        print(f"   YAML: {DATA_YAML_PATH}")

if not _skip_conversion:
  os.makedirs(YOLO_DIR, exist_ok=True)

# ── Auto-detect directory structure ──
# Colab/Kaggle can create: Train/Train/Annotations hoặc Train/Annotations
def find_split_dir(base, split_name):
    """Find the folder containing Annotations yourself & JPEGImages."""
    # Try the nested path with two layers first (Train/Train/)
    nested = os.path.join(base, split_name, split_name)
    if os.path.isdir(os.path.join(nested, 'Annotations')):
        return nested
    # Try a single layer (Train/)
    direct = os.path.join(base, split_name)
    if os.path.isdir(os.path.join(direct, 'Annotations')):
        return direct
    print(f"  ⚠️  No annotations found in {split_name}")
    return None

SPLITS = {}
if not _skip_conversion:
    for name in ['Train', 'Val', 'Test']:
        split_dir = find_split_dir(SOURCE_DIR, name)
        if split_dir:
            SPLITS[name.lower()] = split_dir
            print(f"  ✓ {name:5s} → {split_dir}")

    if not SPLITS:
        raise FileNotFoundError(f"No data found in {SOURCE_DIR}")

def parse_voc_xml(xml_path, fallback_img_path=None):
    """Parse file XML Pascal VOC."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    img_w = int(size.find('width').text) if size is not None and size.find('width') is not None else 0
    img_h = int(size.find('height').text) if size is not None and size.find('height') is not None else 0
    if (img_w <= 0 or img_h <= 0) and fallback_img_path and os.path.exists(fallback_img_path):
        img = cv2.imread(fallback_img_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
    if img_w <= 0 or img_h <= 0:
        return []
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.strip().lower()
        bbox = obj.find('bndbox')
        xmin = max(0, min(float(bbox.find('xmin').text), img_w))
        ymin = max(0, min(float(bbox.find('ymin').text), img_h))
        xmax = max(0, min(float(bbox.find('xmax').text), img_w))
        ymax = max(0, min(float(bbox.find('ymax').text), img_h))
        objects.append({'class_name': class_name, 'xmin': xmin, 'ymin': ymin,
                        'xmax': xmax, 'ymax': ymax, 'img_w': img_w, 'img_h': img_h})
    return objects

def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h

# ── If you haven't converted it yet, complete the entire process ──
if not _skip_conversion:
    # Step 1: Collect class names
    print("\n🔍 Step 1: Scan class names...")
    all_classes = set()
    for split_name, split_dir in SPLITS.items():
        ann_dir = os.path.join(split_dir, "Annotations")
        if not os.path.exists(ann_dir): continue
        for xml_file in glob.glob(os.path.join(ann_dir, "*.xml")):
            for obj in parse_voc_xml(xml_file):
                all_classes.add(obj['class_name'])

    class_list = sorted(all_classes)
    class_to_id = {name: idx for idx, name in enumerate(class_list)}
    print(f"  Classes ({len(class_list)}): {class_list}")
    print(f"  Mapping: {class_to_id}")

    # Step 2: Conversion
    print(f"\n📦 Step 2: Convert VOC → YOLO...")
    stats = {}
    for split_name, split_dir in SPLITS.items():
        ann_dir = os.path.join(split_dir, "Annotations")
        img_dir = os.path.join(split_dir, "JPEGImages")
        if not os.path.exists(ann_dir) or not os.path.exists(img_dir):
            print(f"  ⚠️  Skip {split_name}: missing Annotations or JPEGImages")
            continue
        yolo_img_dir = os.path.join(YOLO_DIR, "images", split_name)
        yolo_lbl_dir = os.path.join(YOLO_DIR, "labels", split_name)
        os.makedirs(yolo_img_dir, exist_ok=True)
        os.makedirs(yolo_lbl_dir, exist_ok=True)
        xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        converted = skipped = total_objects = 0
        for xml_path in xml_files:
            xml_name = os.path.splitext(os.path.basename(xml_path))[0]
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = os.path.join(img_dir, xml_name + ext)
                if os.path.exists(candidate):
                    img_path = candidate; break
            if img_path is None: skipped += 1; continue
            objects = parse_voc_xml(xml_path, fallback_img_path=img_path)
            if not objects: skipped += 1; continue
            img_ext = os.path.splitext(img_path)[1]
            dst_img = os.path.join(yolo_img_dir, xml_name + img_ext)
            if not os.path.exists(dst_img): shutil.copy2(img_path, dst_img)
            label_path = os.path.join(yolo_lbl_dir, xml_name + ".txt")
            with open(label_path, 'w') as f:
                for obj in objects:
                    class_id = class_to_id[obj['class_name']]
                    cx, cy, w, h = voc_to_yolo(obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['img_w'], obj['img_h'])
                    if w > 0 and h > 0:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        total_objects += 1
            converted += 1
        stats[split_name] = {'converted': converted, 'skipped': skipped, 'objects': total_objects}
        print(f"  ✓ {split_name}: {converted} ảnh, {total_objects} objects (skipped: {skipped})")

    # Step 3: Create data.yaml
    data_yaml = {'path': YOLO_DIR, 'train': 'images/train', 'val': 'images/val',
                 'test': 'images/test', 'nc': len(class_list), 'names': class_list}
    DATA_YAML_PATH = os.path.join(YOLO_DIR, "data.yaml")
    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ CONVERSION COMPLETE!")
    print(f"  YAML: {DATA_YAML_PATH}")
    print(f"  Classes ({len(class_list)}): {class_list}")
    print(f"  Stats: {stats}")


# ============================================================================
# CELL 06: EXPERIMENTAL CONFIGURATION
# ============================================================================
OUTPUT_DIR = os.path.join(BASE_DIR, 'yolo_denoise_experiment')
NOISE_LEVELS = [0, 1, 5, 10, 20, 30]

YOLO_MODELS = {
    "yolov8m":  "yolov8m.pt",
    "yolov9m":  "yolov9m.pt",
    "yolov10m": "yolov10m.pt",
    "yolo11m":  "yolo11m.pt",
    "yolo12m":  "yolo12m.pt",
}

DENOISE_METHODS = [
    'noisy',           # No noise cancellation (baseline)
    'gaussian_filter', # Gaussian Filter
    'bm3d',            # BM3D
    'autoencoder',     # Autoencoder
    'dncnn',           # DnCNN
    'cae_pso',         # CAE + PSO
]

DEVICE_ID = 0 if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

POSITIVE_CLASS_ID = 0   # person
NEGATIVE_CLASS_ID = 1   # person-like

# ── NUM_WORKERS & BATCH SIZE: Optimized for A100 ──
NUM_WORKERS = 48
if torch.cuda.is_available():
    _gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if _gpu_mem >= 70:
        YOLO_BATCH = 64;  DENOISE_INFER_BATCH = 256
    elif _gpu_mem >= 35:
        YOLO_BATCH = 48;  DENOISE_INFER_BATCH = 128
    elif _gpu_mem >= 14:
        YOLO_BATCH = 32;  DENOISE_INFER_BATCH = 64
    else:
        YOLO_BATCH = 16;  DENOISE_INFER_BATCH = 32
    print(f"  🚀 Auto-config: YOLO batch={YOLO_BATCH}, Denoise infer batch={DENOISE_INFER_BATCH}")
else:
    YOLO_BATCH = 16;  DENOISE_INFER_BATCH = 32

# ── A fixed directory (used every time it runs) ──
os.makedirs(OUTPUT_DIR, exist_ok=True)
NOISY_DATASET_DIR    = os.path.join(OUTPUT_DIR, "noisy_datasets")
os.makedirs(NOISY_DATASET_DIR, exist_ok=True)
DENOISED_DATASET_DIR = os.path.join(OUTPUT_DIR, "denoised_datasets")
os.makedirs(DENOISED_DATASET_DIR, exist_ok=True)
DENOISE_MODELS_DIR   = os.path.join(OUTPUT_DIR, "denoise_models")
os.makedirs(DENOISE_MODELS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# AUTO-VERSION RESULTS_DIR
# Each time it runs, it automatically creates a new folder: results_summary_version_1, _2, _3 ...
# Never overwrite old results.
# ══════════════════════════════════════════════════════════════════════
_existing_vers = [
    d for d in os.listdir(OUTPUT_DIR)
    if d.startswith('results_summary_version_')
    and os.path.isdir(os.path.join(OUTPUT_DIR, d))
]
if _existing_vers:
    _max_ver = max(int(d.replace('results_summary_version_', '')) for d in _existing_vers)
    RESULTS_VERSION = _max_ver + 1
else:
    RESULTS_VERSION = 1

RESULTS_DIR = os.path.join(OUTPUT_DIR, f"results_summary_version_{RESULTS_VERSION}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Versioned save path on Google Drive ──
_EXPERIMENT_BASE = '/content/drive/MyDrive/DOCTOR_PHD/FINAL PROJECT/04_RESULT_TRAIN_KARTHY/YOLO_Denoise_Experiment_Karthy'
os.makedirs(_EXPERIMENT_BASE, exist_ok=True)
_existing_exp_vers = [
    d for d in os.listdir(_EXPERIMENT_BASE)
    if d.startswith('ver_') and os.path.isdir(os.path.join(_EXPERIMENT_BASE, d))
]
if _existing_exp_vers:
    _max_exp_ver = max(int(v.split('_')[1]) for v in _existing_exp_vers)
    EXPERIMENT_VERSION = _max_exp_ver + 1
else:
    EXPERIMENT_VERSION = 1
LOCAL_SAVE_DIR = os.path.join(_EXPERIMENT_BASE, f'ver_{EXPERIMENT_VERSION}')

print(f"{' ' * 0}{'=' * 60}")
print(f"  📂 BASE_DIR         : {BASE_DIR}")
print(f"  📂 Data YAML        : {DATA_YAML_PATH}")
print(f"  📂 Output Dir       : {OUTPUT_DIR}")
print(f"  📂 Results Dir      : {RESULTS_DIR}  ← version {RESULTS_VERSION}")
print(f"  📂 Drive Save Dir   : {LOCAL_SAVE_DIR}  ← version {EXPERIMENT_VERSION}")
print(f"  🔢 Noise Levels     : {NOISE_LEVELS}")
print(f"  🔧 Denoise Methods  : {DENOISE_METHODS}")
print(f"  🤖 YOLO Models      : {list(YOLO_MODELS.keys())}")
print(f"  💻 Device           : {DEVICE}")
print(f"  🖥️  OS / Workers     : {platform.system()} / {NUM_WORKERS}")
print(f"  📊 Total runs       : {len(YOLO_MODELS)} × {len(NOISE_LEVELS)} × {len(DENOISE_METHODS)} = {len(YOLO_MODELS) * len(NOISE_LEVELS) * len(DENOISE_METHODS)}")
print(f"{' ' * 0}{'=' * 60}")

print(f"\n📁 Directory structure in: {BASE_DIR}")
print(f"  ├── pedestrian-detection/              ← Original dataset (Kaggle)")
print(f"  ├── dataset_yolo/                      ← Dataset YOLO format")
print(f"  ├── yolo_denoise_experiment/")
print(f"  │   ├── noisy_datasets/")
print(f"  │   ├── denoised_datasets/")
print(f"  │   ├── denoise_models/")
print(f"  │   ├── training_runs/")
print(f"  │   ├── results_summary_version_1/     ← Run 1")
print(f"  │   ├── results_summary_version_2/     ← Run 2")
print(f"  │   └── results_summary_version_{RESULTS_VERSION}/  ← This run")
print(f"  └── YOLO_Denoise_Experiment_Karthy/")
print(f"      ├── ver_1/")
print(f"      └── ver_{EXPERIMENT_VERSION}/        ← This drive backup")


# ============================================================================
# CELL 07: GAUSSIAN NOISE ADDING FUNCTION & CREATE DATASET
# ============================================================================
def add_gaussian_noise(image, sigma):
    """Adding Gaussian noise to BGR uint8 images"""
    if sigma == 0:
        return image.copy()
    noise = np.random.normal(0, sigma, image.shape).astype(np.float64)
    noisy_image = image.astype(np.float64) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def create_noisy_dataset(original_yaml_path, noise_sigma, output_base_dir):
    """
    Create a copy of the dataset with Gaussian noise.
    Simply add noise to the image, keep the labels intact.
    """
    if noise_sigma == 0:
        return original_yaml_path

    # ⚡ Skip if you have already created it.
    noisy_root = os.path.join(output_base_dir, f"noise_{noise_sigma}")
    yaml_check = os.path.join(noisy_root, 'data.yaml')
    if os.path.exists(yaml_check):
        _n = len(glob.glob(os.path.join(noisy_root, 'images', 'train', '*')))
        if _n > 0:
            print(f"  ⚡ Noisy dataset σ={noise_sigma} already exists ({_n} images) → ignore")
            return yaml_check

    print(f"\n  Create a dataset with noise σ = {noise_sigma}")

    with open(original_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    original_root = data_config.get('path', os.path.dirname(os.path.abspath(original_yaml_path)))
    noisy_root = os.path.join(output_base_dir, f"noise_{noise_sigma}")

    np.random.seed(noise_sigma)

    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
        src_img_dir = os.path.join(original_root, data_config[split])
        src_lbl_dir = src_img_dir.replace('images', 'labels')
        dst_img_dir = os.path.join(noisy_root, data_config[split])
        dst_lbl_dir = dst_img_dir.replace('images', 'labels')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        img_files = sorted(glob.glob(os.path.join(src_img_dir, '*')))
        for img_path in img_files:
            fname = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None: continue
            noisy = add_gaussian_noise(img, noise_sigma)
            cv2.imwrite(os.path.join(dst_img_dir, fname), noisy)

            # Copy label
            lbl_name = os.path.splitext(fname)[0] + '.txt'
            src_lbl = os.path.join(src_lbl_dir, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(dst_lbl_dir, lbl_name))

        print(f"    {split}: {len(img_files)} image")

    # Create a new YAML
    noisy_yaml = data_config.copy()
    noisy_yaml['path'] = noisy_root
    yaml_path = os.path.join(noisy_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(noisy_yaml, f, default_flow_style=False)

    return yaml_path


print("✅ The noise function is ready")


# ============================================================================
# CELL 08: DEFINITION OF 5 NOISE REDUCTION ALGORITHMS
# ============================================================================
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. GAUSSIAN FILTER (OpenCV)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def denoise_gaussian_filter(image, sigma=None):
    """
    Noise reduction using a Gaussian filter.
    Kernel size is automatically adjusted based on noise level.
    """
    if sigma is None or sigma <= 1:
        ksize = 3
    elif sigma <= 5:
        ksize = 5
    elif sigma <= 15:
        ksize = 7
    else:
        ksize = 9
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. BM3D (Block-Matching 3D)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def denoise_bm3d(image, sigma=10):
    """
    Noise reduction using BM3D.
    Input: BGR uint8, Output: BGR uint8
    """
    # BM3D works on float images [0, 1]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float64) / 255.0

    sigma_psd = sigma / 255.0
    if sigma_psd < 0.01:
        sigma_psd = 0.01  # Minimum sigma

    # BM3D for color images
    denoised = bm3d.bm3d_rgb(img_float, sigma_psd=sigma_psd)
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)


print("✅ Gaussian Filter & BM3D are ready")


# ============================================================================
# CELL 09: AUTOENCODER (Convolutional Denoising Autoencoder)
# ============================================================================
class DenoisingAutoencoder(nn.Module):
    """Convolutional Denoising Autoencoder."""
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class NoisyImageDataset(Dataset):
    """Dataset cho training denoising models: return (noisy, clean) pairs."""
    def __init__(self, image_dir, sigma, patch_size=128, patches_per_image=4):
        self.sigma = sigma
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self.image_paths = sorted(self.image_paths)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img = cv2.imread(self.image_paths[img_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Random crop
        ps = self.patch_size
        if h >= ps and w >= ps:
            y = np.random.randint(0, h - ps + 1)
            x = np.random.randint(0, w - ps + 1)
            clean_patch = img[y:y+ps, x:x+ps]
        else:
            clean_patch = cv2.resize(img, (ps, ps))

        # Add noise
        noise = np.random.normal(0, self.sigma, clean_patch.shape).astype(np.float64)
        noisy_patch = np.clip(clean_patch.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        clean_tensor = self.transform(clean_patch)
        noisy_tensor = self.transform(noisy_patch)

        return noisy_tensor, clean_tensor


def train_autoencoder(train_img_dir, sigma, save_path, epochs=30, batch_size=16, lr=1e-3):
    """Train Denoising Autoencoder."""
    if os.path.exists(save_path):
        print(f"  ⚡ The model already exists: {save_path}")
        model = DenoisingAutoencoder().to(DEVICE)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        return model

    print(f"  🏋️ Training Autoencoder cho σ={sigma}...")
    dataset = NoisyImageDataset(train_img_dir, sigma, patch_size=128, patches_per_image=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model = DenoisingAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')  # 🚀 AMP for A100

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):  # 🚀 AMP
                output = model(noisy)
                loss = criterion(output, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Saved: {save_path}")
    return model


def denoise_with_model(image, model, patch_size=128, stride=96):
    """
    Image noise reduction using deep learning models (sliding window + overlap averaging).
    Input: BGR uint8, Output: BGR uint8
    """
    model.eval()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape

    # Pad to multiples of patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    ph, pw = img_padded.shape[:2]
    result = np.zeros((ph, pw, c), dtype=np.float64)
    count = np.zeros((ph, pw, 1), dtype=np.float64)

    transform = transforms.ToTensor()

    with torch.no_grad():
        for y in range(0, ph - patch_size + 1, stride):
            for x in range(0, pw - patch_size + 1, stride):
                patch = img_padded[y:y+patch_size, x:x+patch_size]
                patch_tensor = transform(patch).unsqueeze(0).to(DEVICE)
                output = model(patch_tensor)
                output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                result[y:y+patch_size, x:x+patch_size] += output_np
                count[y:y+patch_size, x:x+patch_size] += 1

    count = np.maximum(count, 1)
    result = result / count
    result = np.clip(result[:h, :w] * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


print("✅ Autoencoder sẵn sàng")


# ============================================================================
# CELL 10: DnCNN (Denoising Convolutional Neural Network)
# ============================================================================
class DnCNN(nn.Module):
    """
    DnCNN - Zhang et al. "Beyond a Gaussian Denoiser" (2017)
    Residual learning: model học noise residual, output = input - predicted_noise
    """
    def __init__(self, channels=3, num_layers=17, features=64):
        super().__init__()
        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(channels, features, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # Middle layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # Last layer: Conv (predict noise residual)
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.layers(x)
        return x - noise  # Residual learning


def train_dncnn(train_img_dir, sigma, save_path, epochs=30, batch_size=16, lr=1e-3):
    """Train DnCNN."""
    if os.path.exists(save_path):
        print(f"  ⚡ Model đã tồn tại: {save_path}")
        model = DnCNN(channels=3, num_layers=17, features=64).to(DEVICE)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        return model

    print(f"  🏋️ Training DnCNN cho σ={sigma}...")
    dataset = NoisyImageDataset(train_img_dir, sigma, patch_size=128, patches_per_image=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model = DnCNN(channels=3, num_layers=17, features=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')  # 🚀 AMP for A100

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):  # 🚀 AMP
                output = model(noisy)
                loss = criterion(output, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"  ✓ Saved: {save_path}")
    return model


print("✅ DnCNN is ready")


# ============================================================================
# CELL 11: CAE + PSO (Convolutional Autoencoder + Particle Swarm Optimization)
# ============================================================================
#
# CAE: Similar architecture to Autoencoder, but PSO optimizes hyperparameters
# PSO: Optimization (learning_rate, num_filters, kernel_size) để minimize val loss
#
# PSO update rules:
#   v_i(t+1) = w * v_i(t) + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
#   x_i(t+1) = x_i(t) + v_i(t+1)
#
# ============================================================================

class CAE(nn.Module):
    """Convolutional Autoencoder với configurable hyperparameters."""
    def __init__(self, base_filters=64, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        f1 = base_filters
        f2 = base_filters * 2

        self.encoder = nn.Sequential(
            nn.Conv2d(3, f1, kernel_size, padding=pad),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f1, kernel_size, padding=pad),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(f1, f2, kernel_size, padding=pad),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f2, kernel_size, padding=pad),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(f2, f2, 2, stride=2),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f1, kernel_size, padding=pad),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(f1, f1, 2, stride=2),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, 3, kernel_size, padding=pad),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def pso_fitness(params, train_loader, val_loader, epochs=10):
    """
    Fitness function cho PSO: train CAE với params, return to val_loss.
    params: [lr, base_filters, kernel_size]
    """
    lr = params[0]
    base_filters = int(params[1])
    kernel_size = int(params[2])
    if kernel_size % 2 == 0:
        kernel_size += 1

    try:
        model = CAE(base_filters=base_filters, kernel_size=kernel_size).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler('cuda')  # 🚀 AMP
        model.train()
        for epoch in range(epochs):
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(noisy), clean)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
                val_loss += criterion(model(noisy), clean).item()
                count += 1
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return val_loss / max(count, 1)
    except Exception as e:
        print(f"    PSO fitness error: {e}")
        return float('inf')


def particle_swarm_optimization(train_loader, val_loader, n_particles=6, n_iterations=10):
    """
    PSO is used to find optimal hyperparameters for CAE.

    v_i(t+1) = w * v_i(t) + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
    x_i(t+1) = x_i(t) + v_i(t+1)

    w giảm dần 0.9 → 0.4, c1 = c2 = 2.0
    """
    print("  🐝 Running PSO optimization...")

    bounds_low  = np.array([1e-4, 32, 3])
    bounds_high = np.array([5e-3, 96, 7])
    n_dims = len(bounds_low)

    w_start, w_end = 0.9, 0.4
    c1, c2 = 2.0, 2.0
    v_max = (bounds_high - bounds_low) * 0.2
    v_min = -v_max

    particles = np.random.uniform(bounds_low, bounds_high, (n_particles, n_dims))
    velocities = np.random.uniform(v_min, v_max, (n_particles, n_dims))
    fitness = np.full(n_particles, float('inf'))
    pbest = particles.copy()
    pbest_fitness = fitness.copy()
    gbest = particles[0].copy()
    gbest_fitness = float('inf')

    for iteration in range(n_iterations):
        w = w_start - (w_start - w_end) * iteration / max(n_iterations - 1, 1)
        for i in range(n_particles):
            fitness[i] = pso_fitness(particles[i], train_loader, val_loader, epochs=8)
            if fitness[i] < pbest_fitness[i]:
                pbest_fitness[i] = fitness[i]
                pbest[i] = particles[i].copy()
            if fitness[i] < gbest_fitness:
                gbest_fitness = fitness[i]
                gbest = particles[i].copy()

        print(f"    PSO iter {iteration+1}/{n_iterations}: "
              f"w={w:.3f}, best_loss={gbest_fitness:.6f}, "
              f"lr={gbest[0]:.5f}, filters={int(gbest[1])}, ks={int(gbest[2])}")

        for i in range(n_particles):
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            cognitive = c1 * r1 * (pbest[i] - particles[i])
            social = c2 * r2 * (gbest - particles[i])
            velocities[i] = w * velocities[i] + cognitive + social
            velocities[i] = np.clip(velocities[i], v_min, v_max)
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], bounds_low, bounds_high)

    best_params = {
        'lr': gbest[0],
        'base_filters': int(gbest[1]),
        'kernel_size': int(gbest[2]) | 1,
    }
    print(f"  ✓ PSO best params: {best_params} (loss={gbest_fitness:.6f})")
    return best_params



def _infer_cae_params_from_state_dict(state_dict):
    """Derive base_filters and kernel_size from CAE's state_dict"""
    # encoder.0.weight shape: [base_filters, 3, kernel_size, kernel_size]
    w = state_dict.get('encoder.0.weight')
    if w is not None:
        return {'base_filters': w.shape[0], 'kernel_size': w.shape[2]}
    return {'base_filters': 64, 'kernel_size': 3}

def train_cae_pso(train_img_dir, val_img_dir, sigma, save_path, full_epochs=30, batch_size=16):
    """Train CAE with optimal hyperparameters using PSO"""
    if os.path.exists(save_path):
        print(f"  ⚡ Model đã tồn tại: {save_path}")
        # 🔑 Deduce the architecture from state_dict (independent of YAML)
        state_dict = torch.load(save_path, map_location=DEVICE)
        params = _infer_cae_params_from_state_dict(state_dict)
        print(f"     → base_filters={params['base_filters']}, kernel_size={params['kernel_size']}")
        model = CAE(
            base_filters=params['base_filters'],
            kernel_size=params['kernel_size']
        ).to(DEVICE)
        model.load_state_dict(state_dict)
        return model

    print(f"  🏋️ Training CAE+PSO for σ={sigma}...")
    train_dataset = NoisyImageDataset(train_img_dir, sigma, patch_size=128, patches_per_image=2)
    val_dataset = NoisyImageDataset(val_img_dir, sigma, patch_size=128, patches_per_image=2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    best_params = particle_swarm_optimization(
        train_loader, val_loader, n_particles=6, n_iterations=10
    )

    print(f"  🏋️ Full training with best params...")
    model = CAE(
        base_filters=best_params['base_filters'],
        kernel_size=best_params['kernel_size']
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')  # 🚀 AMP

    for epoch in range(full_epochs):
        model.train()
        total_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):  # 🚀 AMP
                loss = criterion(model(noisy), clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{full_epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    params_path = save_path.replace('.pt', '_params.yaml')
    # Convert numpy types to Python native before saving YAML
    params_to_save = {}
    for k, v in best_params.items():
        if hasattr(v, 'item'):  # numpy scalar
            params_to_save[k] = v.item()
        else:
            params_to_save[k] = v
    with open(params_path, 'w') as f:
        yaml.dump(params_to_save, f, default_flow_style=False)
    print(f"  ✓ Saved: {save_path}")
    return model


print("✅ CAE + PSO is ready")


# ============================================================================
# CELL 12: NOISE-FREE DATASET CONSTRUCTOR
# ============================================================================
import multiprocessing as mp
from functools import partial

# ── Number CPU cores cho BM3D/Gaussian parallel ──
N_CPU_WORKERS = max(1, mp.cpu_count() - 1)  # Leave one core for the system
print(f"  💻 CPU cores available: {mp.cpu_count()}, using {N_CPU_WORKERS} workers for BM3D/Gaussian")

def denoise_with_model_fast(image, model, patch_size=128, stride=112, batch_size=None):
    """
    ⚡ Noise reduction using deep learning models - BATCHED inference.
    Combine multiple patches into a batch instead of processing each patch individually.
    """
    if batch_size is None:
        batch_size = globals().get('DENOISE_INFER_BATCH', 64)
    model.eval()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img_padded = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    ph, pw = img_padded.shape[:2]

    result = np.zeros((ph, pw, c), dtype=np.float64)
    count = np.zeros((ph, pw, 1), dtype=np.float64)

    transform = transforms.ToTensor()

    patches = []
    positions = []
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            patch = img_padded[y:y+patch_size, x:x+patch_size]
            patches.append(transform(patch))
            positions.append((y, x))

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, len(patches), batch_size):
            batch_patches = torch.stack(patches[i:i+batch_size]).to(DEVICE)
            batch_outputs = model(batch_patches).cpu().numpy()

            for j, (y, x) in enumerate(positions[i:i+batch_size]):
                output_np = batch_outputs[j].transpose(1, 2, 0)
                result[y:y+patch_size, x:x+patch_size] += output_np
                count[y:y+patch_size, x:x+patch_size] += 1

    count = np.maximum(count, 1)
    result = result / count
    result = np.clip(result[:h, :w] * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 WORKER FUNCTION cho multiprocessing (BM3D / Gaussian Filter)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _denoise_single_image(args):
    """Worker function: đọc ảnh → denoise → ghi ra file. Chạy song song."""
    img_path, dst_img_path, src_lbl, dst_lbl, method, sigma = args

    # Skip if it already exists
    if os.path.exists(dst_img_path):
        if src_lbl and os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        return True

    img = cv2.imread(img_path)
    if img is None:
        return False

    if method == 'gaussian_filter':
        denoised = denoise_gaussian_filter(img, sigma=sigma)
    elif method == 'bm3d':
        denoised = denoise_bm3d(img, sigma=sigma)
    else:
        return False

    cv2.imwrite(dst_img_path, denoised)

    if src_lbl and os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)

    return True


def create_denoised_dataset(
    noisy_yaml_path,
    denoise_method,
    noise_sigma,
    output_base_dir,
    denoise_model=None,
):
    """
    Create a denoised dataset
    ⚡ BM3D & Gaussian Filter: multiprocessing song song trên tất cả CPU cores
    ⚡ Autoencoder/DnCNN/CAE+PSO: GPU batch inference
    ⚡ Skip if it already exists.
    """
    denoised_root = os.path.join(output_base_dir, f"{denoise_method}_noise_{noise_sigma}")
    yaml_path = os.path.join(denoised_root, 'data.yaml')

    # ⚡ Skip if you have already created it.
    if os.path.exists(yaml_path):
        print(f"  ⚡ It already existed: {denoise_method} σ={noise_sigma} → skip")
        return yaml_path

    print(f"\n  📸 Create denoised dataset: {denoise_method} (σ={noise_sigma})")

    with open(noisy_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    noisy_root = data_config.get('path', os.path.dirname(os.path.abspath(noisy_yaml_path)))

    use_multiprocessing = denoise_method in ['gaussian_filter', 'bm3d']

    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue

        src_img_dir = os.path.join(noisy_root, data_config[split])
        src_lbl_dir = src_img_dir.replace('images', 'labels')
        dst_img_dir = os.path.join(denoised_root, data_config[split])
        dst_lbl_dir = dst_img_dir.replace('images', 'labels')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        img_files = sorted(glob.glob(os.path.join(src_img_dir, '*')))
        t0 = time.time()

        if use_multiprocessing:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 🚀 MULTIPROCESSING: BM3D & Gaussian Filter
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            tasks = []
            for img_path in img_files:
                fname = os.path.basename(img_path)
                dst_img_path = os.path.join(dst_img_dir, fname)
                lbl_name = os.path.splitext(fname)[0] + '.txt'
                src_lbl = os.path.join(src_lbl_dir, lbl_name)
                dst_lbl = os.path.join(dst_lbl_dir, lbl_name)
                tasks.append((img_path, dst_img_path, src_lbl, dst_lbl, denoise_method, noise_sigma))

            # Filter out already done
            tasks_todo = [t for t in tasks if not os.path.exists(t[1])]
            tasks_done = len(tasks) - len(tasks_todo)

            if tasks_todo:
                n_workers = min(N_CPU_WORKERS, len(tasks_todo))
                print(f"    {split}: {len(tasks_todo)} ảnh cần xử lý ({tasks_done} đã xong), {n_workers} CPU workers...")

                with mp.Pool(processes=n_workers) as pool:
                    results = []
                    for i, result in enumerate(pool.imap_unordered(_denoise_single_image, tasks_todo)):
                        results.append(result)
                        done = tasks_done + i + 1
                        if (i + 1) % 50 == 0 or (i + 1) == len(tasks_todo):
                            elapsed = time.time() - t0
                            speed = (i + 1) / elapsed
                            remaining = (len(tasks_todo) - i - 1) / max(speed, 0.01)
                            print(f"    {split}: {done}/{len(tasks)} "
                                  f"({speed:.1f} img/s, ~{remaining/60:.1f}m left)    ", end='\r')

                # Copy labels for the skipped images.
                for t in tasks:
                    src_lbl = t[2]
                    dst_lbl = t[3]
                    if src_lbl and os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                        shutil.copy2(src_lbl, dst_lbl)
            else:
                print(f"    {split}: All {len(tasks)} photos are done → skip")

            elapsed = time.time() - t0
            print(f"    {split}: {len(img_files)} ảnh ✓ ({elapsed:.1f}s) [🚀 {N_CPU_WORKERS} workers]")

        else:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # GPU inference: Autoencoder / DnCNN / CAE+PSO (sequential)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            for idx, img_path in enumerate(img_files):
                fname = os.path.basename(img_path)
                dst_img_path = os.path.join(dst_img_dir, fname)

                if os.path.exists(dst_img_path):
                    lbl_name = os.path.splitext(fname)[0] + '.txt'
                    src_lbl = os.path.join(src_lbl_dir, lbl_name)
                    dst_lbl = os.path.join(dst_lbl_dir, lbl_name)
                    if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                        shutil.copy2(src_lbl, dst_lbl)
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    continue

                if denoise_method in ['autoencoder', 'dncnn', 'cae_pso']:
                    denoised = denoise_with_model_fast(img, denoise_model)
                else:
                    denoised = img

                cv2.imwrite(dst_img_path, denoised)

                lbl_name = os.path.splitext(fname)[0] + '.txt'
                src_lbl = os.path.join(src_lbl_dir, lbl_name)
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, os.path.join(dst_lbl_dir, lbl_name))

                if (idx + 1) % 50 == 0 or (idx + 1) == len(img_files):
                    elapsed = time.time() - t0
                    speed = (idx + 1) / elapsed
                    remaining = (len(img_files) - idx - 1) / max(speed, 0.01)
                    print(f"    {split}: {idx+1}/{len(img_files)} "
                          f"({speed:.1f} img/s, ~{remaining/60:.1f}m left)    ", end='\r')

            elapsed = time.time() - t0
            print(f"    {split}: {len(img_files)} ảnh ✓ ({elapsed:.1f}s) [GPU batch]")

    # Create YAML
    denoised_yaml = data_config.copy()
    denoised_yaml['path'] = denoised_root
    with open(yaml_path, 'w') as f:
        yaml.dump(denoised_yaml, f, default_flow_style=False)

    print(f"  ✓ YAML: {yaml_path}")
    return yaml_path


print("✅ The create_denoised_dataset function is ready (⚡ Multiprocessing + GPU Batch)")


# ============================================================================
# CELL 13: TRAIN YOLO FUNCTION & EXTRACT RESULTS
# ============================================================================
def extract_best_metrics(results_csv_path, model_name, noise_sigma, denoise_method):
    """Extract the best metrics from results.csv."""
    if not os.path.exists(results_csv_path):
        return None
    try:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
        metric_mappings = {
            'mAP50': ['metrics/mAP50(B)', 'metrics/mAP_0.5', 'mAP50(B)'],
            'mAP50-95': ['metrics/mAP50-95(B)', 'metrics/mAP_0.5:0.95', 'mAP50-95(B)'],
            'Precision': ['metrics/precision(B)', 'precision(B)'],
            'Recall': ['metrics/recall(B)', 'recall(B)'],
        }
        results = {'model': model_name, 'noise_sigma': noise_sigma, 'denoise_method': denoise_method}
        for metric_key, possible_cols in metric_mappings.items():
            for col in possible_cols:
                if col in df.columns:
                    results[metric_key] = df[col].max()
                    break
            else:
                results[metric_key] = None

        # Composite score
        mAP50 = results.get('mAP50') or 0
        mAP95 = results.get('mAP50-95') or 0
        results['Composite'] = 0.5 * mAP50 + 0.5 * mAP95

        print(f"  📊 mAP50={mAP50:.4f}, mAP50-95={mAP95:.4f}")
        return results
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return None


def train_yolo_model(model_name, model_weights, yaml_file_path, noise_sigma,
                     denoise_method, output_dir, device_id=0):
    """Train a YOLO model with resume epoch support-level."""
    result_folder_name = f"{model_name}_{denoise_method}_noise_{noise_sigma}"
    run_dir = os.path.join(output_dir, result_folder_name)
    result_csv_path = os.path.join(run_dir, "results.csv")
    last_pt_path = os.path.join(run_dir, "weights", "last.pt")

    print(f"\n  TRAINING: {model_name} | {denoise_method} | σ={noise_sigma}")

    # ── Case 1: Training is complete (results.csv with all epochs or early stops is available) ──
    if os.path.exists(result_csv_path):
        try:
            _df = pd.read_csv(result_csv_path)
            _n_epochs = len(_df)
            print(f"  ⚡ Đã có kết quả ({_n_epochs} epochs) → bỏ qua")
            return extract_best_metrics(result_csv_path, model_name, noise_sigma, denoise_method)
        except:
            pass

    # ── Case 2: Training was interrupted (last.pt file exists but is not yet complete) ──
    _resume_mode = False
    if os.path.exists(last_pt_path):
        try:
            _ckpt = torch.load(last_pt_path, map_location='cpu', weights_only=False)
            _epoch_done = _ckpt.get('epoch', 0)
            _best_fitness = _ckpt.get('best_fitness', 0)
            print(f"  🔄 RESUME từ epoch {_epoch_done} (best_fitness={_best_fitness:.4f})")
            _resume_mode = True
            del _ckpt
        except Exception as e:
            print(f"  ⚠ last.pt corrupt ({e}) → train từ đầu")
            _resume_mode = False

    try:
        if _resume_mode:
            # ── Resume: load từ last.pt ──
            model = YOLO(last_pt_path)
            start_time = time.time()
            model.train(resume=True)
        else:
            # ── Train mới từ đầu ──
            model = YOLO(model_weights)
            _batch = YOLO_BATCH

            start_time = time.time()
            model.train(
                data=yaml_file_path, epochs=500, patience=50, batch=_batch, imgsz=640,
                amp=True,  # 🚀 Mixed Precision (Tensor Cores A100)
                device=device_id, workers=NUM_WORKERS, project=output_dir, name=result_folder_name,
                exist_ok=True, pretrained=True,
                optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
                cos_lr=False, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5,
                shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0,
                copy_paste=0.0, close_mosaic=10,
                box=7.5, cls=0.5, dfl=1.5,
                seed=0, deterministic=True, save_json=True, save=True, verbose=True,
            )

        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed/60:.1f} phút")
        metrics = extract_best_metrics(result_csv_path, model_name, noise_sigma, denoise_method)
        del model; gc.collect()
        if str(device_id) != 'cpu': torch.cuda.empty_cache()
        return metrics
    except Exception as e:
        print(f"  ✗ LỖI: {e}")
        import traceback; traceback.print_exc()
        gc.collect()
        if str(device_id) != 'cpu':
            try: torch.cuda.empty_cache()
            except: pass
        return None


print("✅ The YOLO training function is ready (with resume epoch-level)")


# ============================================================================
# CELL 14: STEP 1 - CREATE ALL NOISY DATASETS
# ============================================================================
print("=" * 70)
print("  STEP 1: Create datasets with noise")
print("=" * 70)

noisy_yaml_paths = {}
for sigma in NOISE_LEVELS:
    yaml_path = create_noisy_dataset(DATA_YAML_PATH, sigma, NOISY_DATASET_DIR)
    noisy_yaml_paths[sigma] = yaml_path

print("\n✅ All noisy datasets are ready!")
for sigma, path in noisy_yaml_paths.items():
    print(f"  σ={sigma}: {path}")


# ============================================================================
# CELL 15: BƯỚC 2 - TRAIN DENOISE MODELS (Autoencoder, DnCNN, CAE+PSO)
# ============================================================================
print("=" * 70)
print("  BƯỚC 2: Train các denoise models")
print("=" * 70)

# Đường dẫn train images (original clean)
with open(DATA_YAML_PATH, 'r') as f:
    data_config = yaml.safe_load(f)
dataset_root = data_config.get('path', os.path.dirname(DATA_YAML_PATH))
train_img_dir = os.path.join(dataset_root, data_config['train'])
val_img_dir = os.path.join(dataset_root, data_config['val'])

print(f"  Train images: {train_img_dir}")
print(f"  Val images: {val_img_dir}")

# Store trained models per Sigma
denoise_models = {}  # {(method, sigma): model}

# 🚀 Optimized for A100: larger batch size to take advantage of GPU memory.
if torch.cuda.is_available():
    _gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if _gpu_mem >= 70:    # A100 80GB
        BATCH_SIZE = 256
    elif _gpu_mem >= 35:  # A100 40GB
        BATCH_SIZE = 128
    elif _gpu_mem >= 14:  # T4/V100 16GB
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 32
else:
    BATCH_SIZE = 16
print(f"  🚀 Denoise training BATCH_SIZE = {BATCH_SIZE}")
AE_EPOCHS = 50            # 🚀 Production: 50 epochs for Autoencoder
DNCNN_EPOCHS = 50         # 🚀 Production: 50 epochs for DnCNN
CAE_FULL_EPOCHS = 50      # 🚀 Production: 50 epochs full train after PSO
PSO_PARTICLES = 6         # 🚀 Production: 6 particles (enoght diversity)
PSO_ITERATIONS = 10       # 🚀 Production: 10 iterations (good convergence)
PSO_FIT_EPOCHS = 8        # 🚀 Production: 8 epochs each fitness eval

for sigma in NOISE_LEVELS:
    if sigma == 0:
        continue  # No noise reduction is needed for σ=0

    print(f"\n{'═' * 60}")
    print(f"  Train denoise models for σ={sigma}")
    print(f"{'═' * 60}")

    # Autoencoder
    ae_path = os.path.join(DENOISE_MODELS_DIR, f"autoencoder_sigma{sigma}.pt")
    ae_model = train_autoencoder(train_img_dir, sigma, ae_path,
                                  epochs=AE_EPOCHS, batch_size=BATCH_SIZE)
    denoise_models[('autoencoder', sigma)] = ae_model

    # DnCNN
    dncnn_path = os.path.join(DENOISE_MODELS_DIR, f"dncnn_sigma{sigma}.pt")
    dncnn_model = train_dncnn(train_img_dir, sigma, dncnn_path,
                               epochs=DNCNN_EPOCHS, batch_size=BATCH_SIZE)
    denoise_models[('dncnn', sigma)] = dncnn_model

    # CAE + PSO
    cae_path = os.path.join(DENOISE_MODELS_DIR, f"cae_pso_sigma{sigma}.pt")
    cae_model = train_cae_pso(train_img_dir, val_img_dir, sigma, cae_path,
                               full_epochs=CAE_FULL_EPOCHS, batch_size=BATCH_SIZE)
    denoise_models[('cae_pso', sigma)] = cae_model

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n✅ Đã train {len(denoise_models)} denoise models")


# ============================================================================
# CELL 16: STEP 3 - CREATE ALL DENOISED DATASETS
# ============================================================================
print("=" * 70)
print("  BƯỚC 3: Create denoised datasets")
print("=" * 70)

# ⚡ AUTO-RELOAD: If the denoise_models dict is empty (due to Colab restart),
#    Automatically reload from the previously trained .pt file.
if not denoise_models:
    print("  🔄 Reloading denoise models from disk....")
    _loaded = 0
    for sigma in NOISE_LEVELS:
        if sigma == 0:
            continue
        # Autoencoder
        ae_path = os.path.join(DENOISE_MODELS_DIR, f"autoencoder_sigma{sigma}.pt")
        if os.path.exists(ae_path):
            ae_model = DenoisingAutoencoder().to(DEVICE)
            ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
            ae_model.eval()
            denoise_models[('autoencoder', sigma)] = ae_model
            _loaded += 1
        # DnCNN
        dncnn_path = os.path.join(DENOISE_MODELS_DIR, f"dncnn_sigma{sigma}.pt")
        if os.path.exists(dncnn_path):
            dncnn_model = DnCNN(channels=3, num_layers=17, features=64).to(DEVICE)
            dncnn_model.load_state_dict(torch.load(dncnn_path, map_location=DEVICE))
            dncnn_model.eval()
            denoise_models[('dncnn', sigma)] = dncnn_model
            _loaded += 1
        # CAE+PSO (suy ra architecture từ state_dict)
        cae_path = os.path.join(DENOISE_MODELS_DIR, f"cae_pso_sigma{sigma}.pt")
        if os.path.exists(cae_path):
            cae_ckpt = torch.load(cae_path, map_location=DEVICE)
            _p = _infer_cae_params_from_state_dict(cae_ckpt)
            cae_model = CAE(base_filters=_p['base_filters'], kernel_size=_p['kernel_size']).to(DEVICE)
            cae_model.load_state_dict(cae_ckpt)
            cae_model.eval()
            denoise_models[('cae_pso', sigma)] = cae_model
            _loaded += 1
    print(f"  ✅ Đã reload {_loaded} denoise models từ disk")
    gc.collect(); torch.cuda.empty_cache()

denoised_yaml_paths = {}  # {(method, sigma): yaml_path}

for sigma in NOISE_LEVELS:
    if sigma == 0:
        # σ=0: Use the original dataset for all methods.
        for method in DENOISE_METHODS:
            denoised_yaml_paths[(method, 0)] = DATA_YAML_PATH
        continue

    noisy_yaml = noisy_yaml_paths[sigma]

    # Baseline: noisy (no noise cancellation)
    denoised_yaml_paths[('noisy', sigma)] = noisy_yaml

    # Gaussian Filter
    yaml_path = create_denoised_dataset(
        noisy_yaml, 'gaussian_filter', sigma, DENOISED_DATASET_DIR
    )
    denoised_yaml_paths[('gaussian_filter', sigma)] = yaml_path

    # BM3D
    yaml_path = create_denoised_dataset(
        noisy_yaml, 'bm3d', sigma, DENOISED_DATASET_DIR
    )
    denoised_yaml_paths[('bm3d', sigma)] = yaml_path

    # Autoencoder
    yaml_path = create_denoised_dataset(
        noisy_yaml, 'autoencoder', sigma, DENOISED_DATASET_DIR,
        denoise_model=denoise_models[('autoencoder', sigma)]
    )
    denoised_yaml_paths[('autoencoder', sigma)] = yaml_path

    # DnCNN
    yaml_path = create_denoised_dataset(
        noisy_yaml, 'dncnn', sigma, DENOISED_DATASET_DIR,
        denoise_model=denoise_models[('dncnn', sigma)]
    )
    denoised_yaml_paths[('dncnn', sigma)] = yaml_path

    # CAE + PSO
    yaml_path = create_denoised_dataset(
        noisy_yaml, 'cae_pso', sigma, DENOISED_DATASET_DIR,
        denoise_model=denoise_models[('cae_pso', sigma)]
    )
    denoised_yaml_paths[('cae_pso', sigma)] = yaml_path

    gc.collect()
    torch.cuda.empty_cache()

print(f"\n✅ Total number of denoised datasets: {len(denoised_yaml_paths)}")
for key, path in sorted(denoised_yaml_paths.items()):
    print(f"  {key[0]:>20s} σ={key[1]}: {path}")


# ============================================================================
# CELL 17: TRAINING + RETRAIN + CLASSIFICATION METRICS
# ============================================================================
# A. Training tat ca models x denoise x noise (epochs=500, patience=50)
# B. Retrain cac run bi ngat (< 53 epochs)
# C. Classification metrics (Accuracy, F1, Sensitivity, Specificity)
# ============================================================================

# ══════════════════════════════════════════════════════════════════════
# PART A: TRAINING ALL MODELS x DENOISE x NOISE
# ══════════════════════════════════════════════════════════════════════

TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training_runs")
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)

_ckpt_path = os.path.join(RESULTS_DIR, "results_checkpoint.csv")

# ══════════════════════════════════════════════════════════════════════
# 🔥 FULL PURGE: Delete ALL old results (run only once)
#    After purgeing is complete → create a flag file → next time skip purge (resume)
# ══════════════════════════════════════════════════════════════════════
_PURGE_FLAG = os.path.join(OUTPUT_DIR, "_PURGE_500ep_DONE.flag")

if not os.path.exists(_PURGE_FLAG):
    print("🔥" * 35)
    print("  FULL PURGE: Delete all old results (epochs=100)")
    print("  Retrain all models with epochs=500 and patience=50")
    print("🔥" * 35)

    # 1. DELETE checkpoint CSV
    _old_count = 0
    if os.path.exists(_ckpt_path):
        _old_count = len(pd.read_csv(_ckpt_path))
        os.remove(_ckpt_path)
    print(f"  🗑️ Checkpoint: {_old_count} rows → deleted")

    # 2. Delete ALL training folders
    _purged = 0
    for _folder in glob.glob(os.path.join(TRAIN_OUTPUT_DIR, "*")):
        if os.path.isdir(_folder):
            shutil.rmtree(_folder)
            _purged += 1
    print(f"  🗑️ Training folders: {_purged} runs deleted")

    # 3. Create a flag → the next run will RESUME instead of purgeing again.
    with open(_PURGE_FLAG, 'w') as f:
        f.write(f"Purged at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Old checkpoint rows: {_old_count}\n")
        f.write(f"Old training folders: {_purged}\n")
        f.write(f"New config: epochs=500, patience=50\n")
    print(f"  ✅ Flag created → next run will RESUME, not purge")
    print()
else:
    # Flag exists → I've already purged, I just need to resume.
    with open(_PURGE_FLAG, 'r') as f:
        print(f"  ℹ️ Already purged ({f.readline().strip()}) → RESUME mode")

# ══════════════════════════════════════════════════════════════════════
# ⚡ RESUME: Load checkpoint (if any runs have completed)
# ══════════════════════════════════════════════════════════════════════
all_results = []
_done_keys = set()
if os.path.exists(_ckpt_path):
    _df_ckpt = pd.read_csv(_ckpt_path)
    _df_ckpt = _df_ckpt.drop_duplicates(subset=['model', 'denoise_method', 'noise_sigma'], keep='last')
    all_results = _df_ckpt.to_dict('records')
    for r in all_results:
        _done_keys.add((r['model'], r['denoise_method'], r['noise_sigma']))
    print(f"  🔄 RESUME: {len(all_results)} completed runs")

total_runs = len(YOLO_MODELS) * len(denoised_yaml_paths)
current_run = 0
_skipped = 0
_trained = 0

print("█" * 70)
print(f"  TRAINING: {total_runs} total | {len(_done_keys)} done | {total_runs - len(_done_keys)} remaining")
print(f"  Config: epochs=500, patience=50, workers={NUM_WORKERS}")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("█" * 70)

experiment_start = time.time()

for denoise_method in DENOISE_METHODS:
    for sigma in NOISE_LEVELS:
        key = (denoise_method, sigma)
        if key not in denoised_yaml_paths:
            continue

        yaml_path = denoised_yaml_paths[key]

        for model_name, model_weights in YOLO_MODELS.items():
            current_run += 1

            # ── Skip if already in the checkpoint ──
            _run_key = (model_name, denoise_method, sigma)
            if _run_key in _done_keys:
                _skipped += 1
                continue

            print(f"\n{'▓' * 60}")
            print(f"  RUN {current_run}/{total_runs}: {model_name} × {denoise_method} × σ={sigma}")
            print(f"  Remaining: ~{total_runs - _skipped - _trained - 1}")
            print(f"{'▓' * 60}")

            metrics = train_yolo_model(
                model_name=model_name,
                model_weights=model_weights,
                yaml_file_path=yaml_path,
                noise_sigma=sigma,
                denoise_method=denoise_method,
                output_dir=TRAIN_OUTPUT_DIR,
                device_id=DEVICE_ID,
            )

            if metrics is not None:
                all_results.append(metrics)
                _done_keys.add(_run_key)
                _trained += 1

            # Save checkpoint sau mỗi run
            if all_results:
                df_ckpt = pd.DataFrame(all_results)
                df_ckpt.to_csv(_ckpt_path, index=False)

elapsed = time.time() - experiment_start
print(f"\n{'█' * 70}")
print(f"  ✅ TRAINING COMPLETE")
print(f"  Time: {elapsed/3600:.1f}h | Trained: {_trained} | Skipped: {_skipped}")
print(f"  Total results: {len(all_results)} runs")
print(f"{'█' * 70}")


# ══════════════════════════════════════════════════════════════════════
# PART B: RETRAIN CAC RUN BI NGAT (< 53 EPOCHS)
# ══════════════════════════════════════════════════════════════════════

MIN_VALID_EPOCHS = 53   # 3 warmup + 50 patience = minimum hoi tu

TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training_runs")
_ckpt_path = os.path.join(RESULTS_DIR, "results_checkpoint.csv")

# ── 1. Scan all runs, find runs immediately ──
print("=" * 70)
print("  SCAN: Find intermittent runs (< %d epochs)" % MIN_VALID_EPOCHS)
print("=" * 70)

_incomplete_runs = []
_ok_runs = []
_missing_runs = []

for denoise_method in DENOISE_METHODS:
    for sigma in NOISE_LEVELS:
        for model_name in YOLO_MODELS:
            run_name = f"{model_name}_{denoise_method}_noise_{sigma}"
            run_dir = os.path.join(TRAIN_OUTPUT_DIR, run_name)
            result_csv = os.path.join(run_dir, "results.csv")

            if not os.path.exists(result_csv):
                _missing_runs.append((model_name, denoise_method, sigma, run_name, 0))
                continue

            try:
                _df = pd.read_csv(result_csv)
                n_epochs = len(_df)
            except:
                n_epochs = 0

            if n_epochs < MIN_VALID_EPOCHS:
                _incomplete_runs.append((model_name, denoise_method, sigma, run_name, n_epochs))
            else:
                _ok_runs.append((model_name, denoise_method, sigma, run_name, n_epochs))

print(f"\n  OK (>= {MIN_VALID_EPOCHS} epochs): {len(_ok_runs)} runs")
print(f"  BI NGAT (< {MIN_VALID_EPOCHS} epochs): {len(_incomplete_runs)} runs")
print(f"  Have not results.csv: {len(_missing_runs)} runs")

if _incomplete_runs:
    print(f"\n  List of interrupted runs:")
    for model_name, dm, sig, rn, ne in sorted(_incomplete_runs, key=lambda x: x[4]):
        print(f"    {rn:50s} → {ne:3d} epochs")

# ── 2. Delete the folder and checkpoint of the interrupted runs ──
if not _incomplete_runs:
    print("\n  No tremors were interrupted. All have recovered!")
else:
    print(f"\n  Delete {len(_incomplete_runs)} runs abruptly...")

    # Delete folders
    for model_name, dm, sig, run_name, n_epochs in _incomplete_runs:
        run_dir = os.path.join(TRAIN_OUTPUT_DIR, run_name)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
            print(f"    Deleted: {run_name} ({n_epochs} epochs)")

    # Delete checkpoint CSV
    if os.path.exists(_ckpt_path):
        _df_ckpt = pd.read_csv(_ckpt_path)
        _before = len(_df_ckpt)

        _bad_keys = set()
        for model_name, dm, sig, rn, ne in _incomplete_runs:
            _bad_keys.add((model_name, dm, sig))

        _df_clean = _df_ckpt[~_df_ckpt.apply(
            lambda row: (row['model'], row['denoise_method'], row['noise_sigma']) in _bad_keys,
            axis=1
        )]
        _df_clean.to_csv(_ckpt_path, index=False)
        print(f"\n  Checkpoint: {_before} → {len(_df_clean)} rows ({_before - len(_df_clean)} removed)")

    # ── 3. Retrain ──
    print(f"\n{'=' * 70}")
    print(f"  RETRAIN: {len(_incomplete_runs)} runs (epochs=500, patience=50)")
    print(f"{'=' * 70}")

    # Reload checkpoint cho tracking
    _retrain_results = []
    if os.path.exists(_ckpt_path):
        _retrain_results = pd.read_csv(_ckpt_path).to_dict('records')

    for idx, (model_name, dm, sig, run_name, old_epochs) in enumerate(_incomplete_runs, 1):
        model_weights = YOLO_MODELS[model_name]

        # Tim yaml path
        if sig == 0:
            yaml_path = DATA_YAML_PATH
        elif dm == 'noisy':
            yaml_path = os.path.join(NOISY_DATASET_DIR, f"noise_{sig}", "data.yaml")
        else:
            yaml_path = os.path.join(DENOISED_DATASET_DIR, f"{dm}_noise_{sig}", "data.yaml")

        if not os.path.exists(yaml_path):
            print(f"\n  [{idx}/{len(_incomplete_runs)}] SKIP {run_name} - yaml not found")
            continue

        print(f"\n  [{idx}/{len(_incomplete_runs)}] RETRAIN: {run_name} (was {old_epochs} epochs)")

        metrics = train_yolo_model(
            model_name=model_name,
            model_weights=model_weights,
            yaml_file_path=yaml_path,
            noise_sigma=sig,
            denoise_method=dm,
            output_dir=TRAIN_OUTPUT_DIR,
            device_id=DEVICE_ID,
        )

        if metrics is not None:
            _retrain_results.append(metrics)

            # Verify new epoch count
            _new_csv = os.path.join(TRAIN_OUTPUT_DIR, run_name, "results.csv")
            if os.path.exists(_new_csv):
                _new_epochs = len(pd.read_csv(_new_csv))
                print(f"    {old_epochs} epochs → {_new_epochs} epochs")

        # Save checkpoint sau moi run
        if _retrain_results:
            _df_save = pd.DataFrame(_retrain_results)
            _df_save = _df_save.drop_duplicates(
                subset=['model', 'denoise_method', 'noise_sigma'], keep='last'
            )
            _df_save.to_csv(_ckpt_path, index=False)

    # ── 4. Ket qua ──
    print(f"\n{'=' * 70}")
    print(f"  RETRAIN COMPLETE!")
    print(f"  Retrained: {len(_incomplete_runs)} runs")
    print(f"{'=' * 70}")

    # Verify
    print(f"\n  Verification:")
    _still_bad = 0
    for model_name, dm, sig, run_name, old_epochs in _incomplete_runs:
        _csv = os.path.join(TRAIN_OUTPUT_DIR, run_name, "results.csv")
        if os.path.exists(_csv):
            _ne = len(pd.read_csv(_csv))
            status = "OK" if _ne >= MIN_VALID_EPOCHS else "STILL BAD"
            if _ne < MIN_VALID_EPOCHS:
                _still_bad += 1
            print(f"    {run_name:50s} → {_ne:3d} epochs [{status}]")
        else:
            print(f"    {run_name:50s} → MISSING")
            _still_bad += 1

    if _still_bad == 0:
        print(f"\n  Tat ca runs da hoi tu!")
    else:
        print(f"\n  Con {_still_bad} runs chua dat. Chay lai cell nay.")


# ══════════════════════════════════════════════════════════════════════
# PART C: CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════════════

_possible_dirs = [
    os.path.join(OUTPUT_DIR, "training_runs"),     # Drive path
    '/content/training_runs_local',                 # Local SSD path
]
TRAIN_OUTPUT_DIR = None
for _d in _possible_dirs:
    if os.path.exists(_d) and len(os.listdir(_d)) > 1:
        TRAIN_OUTPUT_DIR = _d
        break
if TRAIN_OUTPUT_DIR is None:
    TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training_runs")

_cls_ckpt_path = os.path.join(RESULTS_DIR, "classification_metrics.csv")

print(f"  TRAIN_OUTPUT_DIR: {TRAIN_OUTPUT_DIR}")
# The number of best.pts exists.
_n_best = sum(1 for d in glob.glob(os.path.join(TRAIN_OUTPUT_DIR, "*"))
              if os.path.exists(os.path.join(d, "weights", "best.pt")))
print(f"  best.pt found: {_n_best}")

def _get_eval_yaml(denoise_method, sigma):
    if sigma == 0:
        return DATA_YAML_PATH
    if denoise_method == 'noisy':
        return os.path.join(NOISY_DATASET_DIR, f"noise_{sigma}", "data.yaml")
    return os.path.join(DENOISED_DATASET_DIR, f"{denoise_method}_noise_{sigma}", "data.yaml")


def compute_classification_from_val(model_path, yaml_path):
    """Load model, val(), trich confusion matrix → TP/FP/FN/TN."""
    _model = YOLO(model_path)
    _val = _model.val(
        data=yaml_path,
        conf=0.25,
        iou=0.5,
        plots=True,
        verbose=False,
        device=DEVICE_ID,
        workers=NUM_WORKERS,
    )

    # Lay confusion matrix
    cm = _val.confusion_matrix.matrix  # shape (nc+1, nc+1)

    # Debug: in shape va gia tri
    print(f"    CM shape={cm.shape}, dtype={cm.dtype}")
    print(f"    CM sum={cm.sum():.1f}, max={cm.max():.1f}")

    # Ultralytics CM: matrix[i][j] = so luong objects cua class i duoc predict la class j
    # Row cuoi = background FP, Col cuoi = background FN (missed)
    # Voi nc=2: shape (3,3)
    #   [0,0]=person->person  [0,1]=person->person-like  [0,2]=person->missed
    #   [1,0]=pers-like->person [1,1]=pers-like->pers-like [1,2]=pers-like->missed
    #   [2,0]=bg->person      [2,1]=bg->person-like      [2,2]=bg->bg

    nc = cm.shape[0] - 1  # so class thuc (khong tinh background)

    if nc < 2:
        # There is only 1 class: person only (no negative class)
        # TP = person detected, FN = person missed, FP = background detected as person
        TP = int(round(cm[0, 0]))
        FN = int(round(cm[0, 1]))     # missed
        FP = int(round(cm[1, 0]))     # background -> person
        TN = 0  # Khong co negative class
    else:
        # 2 classes: person (0), person-like (1)
        # Positive = person (class 0)
        TP = int(round(cm[0, 0]))                             # person -> detected as person
        FN = int(round(cm[0, 1:].sum()))                      # person -> missed hoac nham class khac
        FP = int(round(cm[1, 0])) + int(round(cm[2, 0]))     # person-like + background -> nham la person
        TN = int(round(cm[1, 1])) + int(round(cm[1, 2]))     # person-like -> dung la khong phai person

    del _model
    gc.collect()
    torch.cuda.empty_cache()

    return TP, FP, FN, TN


# ── Delete the old checkpoint (completely zero) and run again ──
if os.path.exists(_cls_ckpt_path):
    _old_df = pd.read_csv(_cls_ckpt_path)
    _all_zero = (_old_df[['TP', 'FP', 'FN', 'TN']].sum().sum() == 0)
    if _all_zero:
        os.remove(_cls_ckpt_path)
        print("  Delete the old checkpoint (completely zero) → restart from the beginning")

# ── Resume ──
_cls_results = []
_cls_done_keys = set()

if os.path.exists(_cls_ckpt_path):
    _cls_df = pd.read_csv(_cls_ckpt_path)
    _cls_df = _cls_df.drop_duplicates(
        subset=['model', 'denoise_method', 'noise_sigma'], keep='last'
    )
    # Chi giu nhung row co du lieu that (khong toan zero)
    _cls_df = _cls_df[(_cls_df['TP'] + _cls_df['FP'] + _cls_df['FN'] + _cls_df['TN']) > 0]
    _cls_results = _cls_df.to_dict('records')
    for r in _cls_results:
        _cls_done_keys.add((r['model'], r['denoise_method'], int(r['noise_sigma'])))
    print(f"  RESUME: {len(_cls_results)} valid classification results")

# ── Build eval list ──
_total = 0
_skip = 0
_no_best = 0
_eval_list = []

for denoise_method in DENOISE_METHODS:
    for sigma in NOISE_LEVELS:
        for model_name in YOLO_MODELS:
            _total += 1
            run_name = f"{model_name}_{denoise_method}_noise_{sigma}"
            best_pt = os.path.join(TRAIN_OUTPUT_DIR, run_name, "weights", "best.pt")

            if (model_name, denoise_method, sigma) in _cls_done_keys:
                _skip += 1
                continue

            if not os.path.exists(best_pt):
                _no_best += 1
                continue

            yaml_path = _get_eval_yaml(denoise_method, sigma)
            if not os.path.exists(yaml_path):
                continue

            _eval_list.append((model_name, denoise_method, sigma, best_pt, yaml_path))

print("=" * 70)
print("  CLASSIFICATION METRICS EVALUATION")
print(f"  Total: {_total} | Done: {_skip} | No best.pt: {_no_best} | To evaluate: {len(_eval_list)}")
print("=" * 70)

if _no_best > 0 and len(_eval_list) == 0 and _skip == 0:
    print(f"\n  KHONG TIM THAY best.pt tai: {TRAIN_OUTPUT_DIR}")
    print("  Kiem tra lai TRAIN_OUTPUT_DIR hoac chay Cell 16 truoc!")
    # List cac folder thuc te
    _dirs = sorted(glob.glob(os.path.join(TRAIN_OUTPUT_DIR, "*")))[:5]
    for d in _dirs:
        _bp = os.path.exists(os.path.join(d, "weights", "best.pt"))
        print(f"    {os.path.basename(d):50s} best.pt={'YES' if _bp else 'NO'}")

# ── Evaluate ──
for idx, (model_name, denoise_method, sigma, best_pt, yaml_path) in enumerate(_eval_list, 1):
    run_name = f"{model_name}_{denoise_method}_noise_{sigma}"
    print(f"\n  [{idx}/{len(_eval_list)}] {run_name}")

    try:
        TP, FP, FN, TN = compute_classification_from_val(best_pt, yaml_path)

        _total_cnt = TP + FP + FN + TN
        Accuracy    = (TP + TN) / _total_cnt if _total_cnt > 0 else 0
        Sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        Precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1_score    = (2 * Precision * Sensitivity /
                       (Precision + Sensitivity)) if (Precision + Sensitivity) > 0 else 0

        row = {
            'model': model_name,
            'noise_sigma': sigma,
            'denoise_method': denoise_method,
            'Accuracy':    round(Accuracy, 4),
            'F1-Score':    round(F1_score, 4),
            'Sensitivity': round(Sensitivity, 4),
            'Specificity': round(Specificity, 4),
            'Precision':   round(Precision, 4),
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        }

        _cls_results.append(row)
        _cls_done_keys.add((model_name, denoise_method, sigma))

        print(f"    Acc={Accuracy:.4f}  F1={F1_score:.4f}  "
              f"Sens={Sensitivity:.4f}  Spec={Specificity:.4f}  "
              f"TP={TP} FP={FP} FN={FN} TN={TN}")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback; traceback.print_exc()

    # Save checkpoint moi 5 runs
    if _cls_results and idx % 5 == 0:
        pd.DataFrame(_cls_results).to_csv(_cls_ckpt_path, index=False)

# ── Save final ──
if _cls_results:
    df_cls = pd.DataFrame(_cls_results)
    df_cls = df_cls.drop_duplicates(
        subset=['model', 'denoise_method', 'noise_sigma'], keep='last'
    )

    df_cls.to_csv(_cls_ckpt_path, index=False)

    _xlsx_path = os.path.join(RESULTS_DIR, "classification_metrics.xlsx")
    df_cls.to_excel(_xlsx_path, index=False, sheet_name='Classification Metrics')

    _xlsx_multi = os.path.join(RESULTS_DIR, "classification_metrics_by_method.xlsx")
    with pd.ExcelWriter(_xlsx_multi, engine='openpyxl') as writer:
        for method in DENOISE_METHODS:
            _df_m = df_cls[df_cls['denoise_method'] == method].sort_values(
                ['model', 'noise_sigma']
            ).reset_index(drop=True)
            if len(_df_m) > 0:
                _df_m.to_excel(writer, sheet_name=method, index=False)

    print(f"\n{'=' * 70}")
    print(f"  CLASSIFICATION METRICS COMPLETE: {len(df_cls)} runs")
    print(f"  CSV:   {_cls_ckpt_path}")
    print(f"  Excel: {_xlsx_path}")
    print(f"  Excel (by method): {_xlsx_multi}")
    print(f"{'=' * 70}")

    # Verify: co row nao zero khong?
    _zero_rows = df_cls[(df_cls['TP'] + df_cls['FP'] + df_cls['FN'] + df_cls['TN']) == 0]
    if len(_zero_rows) > 0:
        print(f"\n  CANH BAO: {len(_zero_rows)} rows co TP=FP=FN=TN=0 (co the bi loi)")
    else:
        print(f"\n  Tat ca {len(df_cls)} rows co du lieu hop le!")
else:
    print("\n  No best.pt available -> run Cell 16 (training) first!")



# ============================================================================
# CELL 18: ASSESSMENT BASED ON ORIGIN (CLEAN) DATA - ALL METRICS
# ============================================================================
# Get the result from σ=0 + noisy (= original data, not much, no noise)
# Combination: mAP50, mAP50-95, Precision, Recall + Accuracy, F1, Sensitivity, Specificity
# Export: CSV, Excel, comparison chart
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Find TRAIN_OUTPUT_DIR ──
_possible_dirs = [
    os.path.join(OUTPUT_DIR, "training_runs"),
    '/content/training_runs_local',
]
TRAIN_OUTPUT_DIR = None
for _d in _possible_dirs:
    if os.path.exists(_d) and len(os.listdir(_d)) > 1:
        TRAIN_OUTPUT_DIR = _d
        break
if TRAIN_OUTPUT_DIR is None:
    TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training_runs")

print(f"  TRAIN_OUTPUT_DIR: {TRAIN_OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════════════
# 1. Get DETECTION METRICS from results_checkpoint.csv
# ══════════════════════════════════════════════════════════════════════
_ckpt_path = os.path.join(RESULTS_DIR, "results_checkpoint.csv")
if not os.path.exists(_ckpt_path):
    # Thu tim o results_checkpoint_full.csv
    _ckpt_path = os.path.join(RESULTS_DIR, "results_checkpoint_full.csv")

assert os.path.exists(_ckpt_path), f"Not found checkpoint: {_ckpt_path}"

df_ckpt = pd.read_csv(_ckpt_path)
df_ckpt.columns = df_ckpt.columns.str.strip()

# Filter: chi lay sigma=0 + noisy (= data goc)
df_origin = df_ckpt[
    (df_ckpt['noise_sigma'] == 0) & (df_ckpt['denoise_method'] == 'noisy')
].copy()

print(f"  Detection metrics (sigma=0, noisy): {len(df_origin)} models")
if len(df_origin) > 0:
    print(df_origin[['model', 'mAP50', 'mAP50-95', 'Precision', 'Recall']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# 2. Count CLASSIFICATION METRICS by model.val() with plots=True
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  CLASSIFICATION METRICS ON DATA ORIGIN")
print(f"{'=' * 70}")

_origin_results = []

for model_name in YOLO_MODELS:
    run_name = f"{model_name}_noisy_noise_0"
    best_pt = os.path.join(TRAIN_OUTPUT_DIR, run_name, "weights", "best.pt")

    if not os.path.exists(best_pt):
        print(f"  {model_name}: best.pt DOES NOT EXIST -> skip")
        continue

    print(f"  {model_name}...", end=" ", flush=True)

    try:
        _model = YOLO(best_pt)
        _val = _model.val(
            data=DATA_YAML_PATH,
            conf=0.25,
            iou=0.5,
            plots=True,
            verbose=False,
            device=DEVICE_ID,
            workers=NUM_WORKERS,
        )

        # Detection metrics
        mAP50 = float(_val.results_dict.get('metrics/mAP50(B)', 0))
        mAP50_95 = float(_val.results_dict.get('metrics/mAP50-95(B)', 0))
        precision_det = float(_val.results_dict.get('metrics/precision(B)', 0))
        recall_det = float(_val.results_dict.get('metrics/recall(B)', 0))

        # Confusion matrix -> classification metrics
        cm = _val.confusion_matrix.matrix
        nc = cm.shape[0] - 1

        if nc < 2:
            TP = int(round(cm[0, 0]))
            FN = int(round(cm[0, 1]))
            FP = int(round(cm[1, 0]))
            TN = 0
        else:
            TP = int(round(cm[0, 0]))
            FN = int(round(cm[0, 1:].sum()))
            FP = int(round(cm[1, 0])) + int(round(cm[2, 0]))
            TN = int(round(cm[1, 1])) + int(round(cm[1, 2]))

        total = TP + FP + FN + TN
        Accuracy    = (TP + TN) / total if total > 0 else 0
        Sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        Specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        Precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1_score    = (2 * Precision * Sensitivity /
                       (Precision + Sensitivity)) if (Precision + Sensitivity) > 0 else 0

        row = {
            'model': model_name,
            'mAP50': round(mAP50, 4),
            'mAP50-95': round(mAP50_95, 4),
            'Accuracy': round(Accuracy, 4),
            'F1-Score': round(F1_score, 4),
            'Sensitivity': round(Sensitivity, 4),
            'Specificity': round(Specificity, 4),
            'Precision': round(precision_det, 4),  # detection Precision từ metrics/precision(B)
            'Recall': round(recall_det, 4),         # detection Recall từ metrics/recall(B)
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        }
        _origin_results.append(row)

        print(f"mAP50={mAP50:.4f}  Acc={Accuracy:.4f}  F1={F1_score:.4f}  "
              f"TP={TP} FP={FP} FN={FN} TN={TN}")

        del _model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# 3. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════
if _origin_results:
    df_origin_full = pd.DataFrame(_origin_results)

    _csv_path = os.path.join(RESULTS_DIR, "origin_baseline_metrics.csv")
    df_origin_full.to_csv(_csv_path, index=False)

    _xlsx_path = os.path.join(RESULTS_DIR, "origin_baseline_metrics.xlsx")
    df_origin_full.to_excel(_xlsx_path, index=False, sheet_name='Origin Baseline')

    print(f"\n{'=' * 70}")
    print("  ORIGIN BASELINE - TAT CA METRICS")
    print(f"{'=' * 70}")
    print(df_origin_full.to_string(index=False))
    print(f"\n  CSV:   {_csv_path}")
    print(f"  Excel: {_xlsx_path}")

    # ══════════════════════════════════════════════════════════════════
    # 4. Comparison Chart Between Models on Data Origin
    # ══════════════════════════════════════════════════════════════════

    # Colors for each model
    _MODEL_COLORS = {
        'yolov8m': '#E91E63', 'yolov9m': '#9C27B0',
        'yolov10m': '#2196F3', 'yolo11m': '#4CAF50', 'yolo12m': '#FF9800',
    }

    models = df_origin_full['model'].tolist()
    colors = [_MODEL_COLORS.get(m, '#888888') for m in models]
    x = np.arange(len(models))
    bar_width = 0.6

    # ── Fig 1: Detection Metrics (mAP50, mAP50-95) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Detection Metrics on Original Data (Clean, No Noise)',
                 fontsize=14, fontweight='bold')

    for ax, metric, title in [
        (axes[0], 'mAP50', 'mAP@50 (%)'),
        (axes[1], 'mAP50-95', 'mAP@50-95 (%)'),
    ]:
        values = df_origin_full[metric].values * 100
        bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha='right', fontsize=9)
        ax.set_ylabel('%')
        ax.set_ylim(min(values) - 3, max(values) + 3)
        ax.grid(axis='y', alpha=0.0)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    _save = os.path.join(RESULTS_DIR, 'origin_detection_metrics.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

    # ── Fig 2: Classification Metrics (Accuracy, F1, Sensitivity, Specificity, Precision) ──
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle('Classification Metrics on Original Data (Clean, No Noise)',
                 fontsize=14, fontweight='bold')

    cls_metrics = [
        ('Accuracy', 'Accuracy (%)'),
        ('F1-Score', 'F1-Score (%)'),
        ('Sensitivity', 'Sensitivity (%)'),
        ('Specificity', 'Specificity (%)'),
        ('Precision', 'Precision (%)'),
    ]

    for ax, (metric, title) in zip(axes, cls_metrics):
        values = df_origin_full[metric].values * 100
        bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('%')
        if max(values) > 0:
            ax.set_ylim(min(values) - 5, min(max(values) + 5, 105))
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    _save = os.path.join(RESULTS_DIR, 'origin_classification_metrics.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

    # ── Fig 3: Radar Chart - Tong hop tat ca metrics ──
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle('Model Comparison on Original Data - Radar Chart',
                 fontsize=14, fontweight='bold', y=1.02)

    radar_metrics = ['mAP50', 'mAP50-95', 'Accuracy', 'F1-Score', 'Sensitivity', 'Specificity', 'Precision']
    N = len(radar_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    for _, row in df_origin_full.iterrows():
        values = [row[m] * 100 for m in radar_metrics]
        values += values[:1]
        color = _MODEL_COLORS.get(row['model'], '#888888')
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save = os.path.join(RESULTS_DIR, 'origin_radar_chart.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

    # ── Fig 4: Grouped Bar - Tat ca metrics 1 bieu do ──
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle('All Metrics Comparison on Original Data', fontsize=14, fontweight='bold')

    all_metrics = ['mAP50', 'mAP50-95', 'Accuracy', 'F1-Score', 'Sensitivity', 'Specificity', 'Precision']
    n_models = len(models)
    n_metrics = len(all_metrics)
    group_width = 0.8
    bw = group_width / n_models
    x_metrics = np.arange(n_metrics)

    for idx, (_, row) in enumerate(df_origin_full.iterrows()):
        offset = (idx - n_models / 2 + 0.5) * bw
        values = [row[m] * 100 for m in all_metrics]
        color = _MODEL_COLORS.get(row['model'], '#888888')
        bars = ax.bar(x_metrics + offset, values, width=bw, label=row['model'],
                      color=color, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x_metrics)
    ax.set_xticklabels(all_metrics, fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    _save = os.path.join(RESULTS_DIR, 'origin_all_metrics_grouped.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

    print(f"\n  TOTAL: 4 documents + 1 CSV + 1 Excel saved to {RESULTS_DIR}")
else:
    print("\n  NO RESULTS - check best.pt at:")
    for m in YOLO_MODELS:
        bp = os.path.join(TRAIN_OUTPUT_DIR, f"{m}_noisy_noise_0", "weights", "best.pt")
        print(f"    {m}: {'YES' if os.path.exists(bp) else 'NO'} - {bp}")



# ============================================================================
# CELL 19: DRAW THE ORIGINAL DATA CHART (Figure 3 & Figure 4)
# ============================================================================
# Data source:
#   - origin_baseline_metrics.csv  → Accuracy, F1-Score, Sensitivity,
#                                     Specificity, mAP50, mAP50-95, Recall
#   - results_checkpoint.csv       → Precision (detection, metrics/precision(B))
#     filter: noise_sigma=0, denoise_method='noisy'
#
# Reasons for merging: Cell 18 saves classification Precision (TP/(TP+FP) from confusion
# matrix) It goes to origin_baseline_metrics.csv, but the paper uses detection.
# Precision (metrics/precision(B) from YOLO val) → need to retrieve from checkpoint.
#
# Output: RESULTS_DIR/paper_figures/
#   Figure3_bar_original_data.png  ← Bar chart 4 detection metrics
#   Figure4_heatmap_original_data.png  ← Heatmap 4 metrics
#   Figure4b_heatmap_all_metrics.png   ← Heatmap 8 metrics (bonus)
# ============================================================================

import matplotlib
matplotlib.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, glob, warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# 0. LOAD DATA
# ──────────────────────────────────────────────────────────────────────
_STD_MODELS = ['yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m']

# --- 0a. origin_baseline_metrics.csv (Accuracy, F1, Sensitivity, Specificity, mAP, Recall) ---
_origin_csv_paths = [
    os.path.join(RESULTS_DIR, "origin_baseline_metrics.csv"),
    os.path.join(OUTPUT_DIR, "results_summary", "origin_baseline_metrics.csv"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo", "origin_baseline_metrics.csv"),
]
df_orig = None
for p in _origin_csv_paths:
    if os.path.exists(p):
        df_orig = pd.read_csv(p)
        df_orig.columns = df_orig.columns.str.strip()
        print(f"  ✅ Origin metrics  : {p}")
        break
if df_orig is None:
    raise FileNotFoundError(
        "❌ Not found origin_baseline_metrics.csv!\n"
        "   Run the cell 'EVALUATION ON DATA ORIGIN' first"
    )

# --- 0b. results_checkpoint.csv no longer needed ---
# Cell 18 has been corrected to save the correct Precision detection (precision_det)
# Go to origin_baseline_metrics.csv → no need to override from checkpoint.
df_ckpt = None

# ──────────────────────────────────────────────────────────────────────
# 1. ARRANGING MODELS
# Cell 18 has been corrected: correct Precision detection is saved (precision_det)
# from metrics/precision(B) instead of classification Precision.
# All columns in origin_baseline_metrics.csv are correct; no override is needed.
# ──────────────────────────────────────────────────────────────────────
df_orig = df_orig[df_orig['model'].isin(_STD_MODELS)].copy()
df_orig['model'] = pd.Categorical(df_orig['model'], categories=_STD_MODELS, ordered=True)
df_orig = df_orig.sort_values('model').reset_index(drop=True)
df_orig['model'] = df_orig['model'].astype(str)  # Remove Category to avoid errors when multiplying numbers.

print(f"\n  📊 Data from origin_baseline_metrics.csv (used to plot the chart):")
_show_cols = ['model', 'mAP50', 'mAP50-95', 'Precision', 'Recall',
              'Accuracy', 'F1-Score', 'Sensitivity', 'Specificity']
_show_cols = [c for c in _show_cols if c in df_orig.columns]
print(df_orig[_show_cols].to_string(index=False))

# ──────────────────────────────────────────────────────────────────────
# COLOR CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
_MODEL_COLORS = {
    'yolov8m':  '#E91E63',
    'yolov9m':  '#9C27B0',
    'yolov10m': '#2196F3',
    'yolo11m':  '#4CAF50',
    'yolo12m':  '#FF9800',
}

PAPER_FIG_DIR = os.path.join(RESULTS_DIR, "paper_figures")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

models    = df_orig['model'].tolist()
colors    = [_MODEL_COLORS.get(m, '#888') for m in models]
x         = np.arange(len(models))
BAR_WIDTH = 0.6

# ──────────────────────────────────────────────────────────────────────
# FIGURE 3: BAR CHART — mAP50 / mAP50-95 / Precision / Recall
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  VẼ FIGURE 3: Bar Chart - Detection Metrics on Original Data")
print(f"{'=' * 70}")

_det_metrics = [
    ('mAP50',    'mAP@50 (%)'),
    ('mAP50-95', 'mAP@50-95 (%)'),
    ('Precision','Precision (%)'),
    ('Recall',   'Recall (%)'),
]

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle('Detection Metrics on Original Data (Clean, No Noise)',
              fontsize=20, fontweight='bold')

for ax, (metric, title) in zip(axes3.flat, _det_metrics):
    if metric not in df_orig.columns:
        ax.set_visible(False)
        continue
    values = df_orig[metric].astype(float).values * 100
    bars = ax.bar(x, values, width=BAR_WIDTH, color=colors,
                  edgecolor='white', linewidth=0.8)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center', fontsize=12)
    ax.set_ylabel('%', fontsize=14)
    ax.set_ylim(max(min(values) - 3, 0), min(max(values) + 3, 100))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
_fig3_path = os.path.join(PAPER_FIG_DIR, 'Figure3_bar_original_data.png')
plt.savefig(_fig3_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✅ Figure 3 saved: {_fig3_path}")

# ──────────────────────────────────────────────────────────────────────
# FIGURE 4: HEATMAP — 4 detection metrics
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  VẼ FIGURE 4: Heatmap - Detection Metrics on Original Data")
print(f"{'=' * 70}")

_hm_metrics    = [m for m in ['mAP50','mAP50-95','Precision','Recall'] if m in df_orig.columns]
_hm_col_labels = {'mAP50':'mAP@50 (%)','mAP50-95':'mAP@50-95 (%)','Precision':'Precision (%)','Recall':'Recall (%)'}

hm_data = df_orig.set_index('model')[_hm_metrics].reindex(_STD_MODELS).astype(float) * 100

fig4, ax4 = plt.subplots(figsize=(14, 6))
fig4.suptitle('Heatmap: All Metrics on Original Data (Clean)',
              fontsize=20, fontweight='bold')

sns.heatmap(
    hm_data,
    annot=True, fmt='.1f',
    cmap='YlOrRd',
    linewidths=1, linecolor='white',
    annot_kws={'size': 14, 'fontweight': 'bold'},
    ax=ax4,
    cbar_kws={'label': 'Score (%)', 'shrink': 0.8},
)
ax4.set_xticklabels([_hm_col_labels.get(m, m) for m in _hm_metrics],
                    rotation=0, ha='center', fontsize=14)
ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=14)
ax4.set_ylabel('')
plt.tight_layout()
_fig4_path = os.path.join(PAPER_FIG_DIR, 'Figure4_heatmap_original_data.png')
plt.savefig(_fig4_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  ✅ Figure 4 saved: {_fig4_path}")

# ──────────────────────────────────────────────────────────────────────
# FIGURE 4b (BONUS): HEATMAP — 8 metrics đầy đủ
# ──────────────────────────────────────────────────────────────────────
_all_metrics = ['mAP50','mAP50-95','Accuracy','F1-Score','Sensitivity','Specificity','Precision','Recall']
_all_metrics = [m for m in _all_metrics if m in df_orig.columns]

if len(_all_metrics) >= 5:
    hm_full = df_orig.set_index('model')[_all_metrics].reindex(_STD_MODELS).astype(float) * 100
    fig4b, ax4b = plt.subplots(figsize=(20, 6))
    fig4b.suptitle('Heatmap: All 8 Metrics on Original Data (Clean)',
                   fontsize=20, fontweight='bold')
    sns.heatmap(
        hm_full,
        annot=True, fmt='.1f',
        cmap='YlOrRd',
        linewidths=1, linecolor='white',
        annot_kws={'size': 13, 'fontweight': 'bold'},
        ax=ax4b,
        cbar_kws={'label': 'Score (%)', 'shrink': 0.8},
    )
    ax4b.set_xticklabels(_all_metrics, rotation=0, ha='right', fontsize=13)
    ax4b.set_yticklabels(ax4b.get_yticklabels(), rotation=0, fontsize=13)
    ax4b.set_ylabel('')
    plt.tight_layout()
    _fig4b_path = os.path.join(PAPER_FIG_DIR, 'Figure4b_heatmap_all_metrics.png')
    plt.savefig(_fig4b_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  ✅ Figure 4b saved: {_fig4b_path}")


# ──────────────────────────────────────────────────────────────────────
# FIGURE 6: LINE CHART — 4 metrics theo noise level (noisy baseline)
# ──────────────────────────────────────────────────────────────────────
SEP = "=" * 70
print(f"\n{SEP}")
print("  VẼ FIGURE 6: Line Chart - Noisy Baseline")
print(SEP)

_ckpt_paths = [
    os.path.join(RESULTS_DIR, "results_checkpoint.csv"),
    os.path.join(RESULTS_DIR, "results_checkpoint_full.csv"),
    os.path.join(OUTPUT_DIR, "results_summary", "results_checkpoint.csv"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo", "results_checkpoint.csv"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo", "results_checkpoint_full.csv"),
]
_df_ckpt = None
for _p in _ckpt_paths:
    if os.path.exists(_p):
        _df_ckpt = pd.read_csv(_p)
        _df_ckpt.columns = _df_ckpt.columns.str.strip()
        print(f"  Loaded: {_p}")
        break

if _df_ckpt is not None:
    _df_noisy_base = _df_ckpt[
        (_df_ckpt["denoise_method"] == "noisy") &
        (_df_ckpt["model"].isin(_STD_MODELS))
    ].copy()

    _SIGMAS = [0, 1, 5, 10, 20, 30]
    _LINE_METRICS = [
        ("mAP50",    "mAP@50 (%)"),
        ("mAP50-95", "mAP@50-95 (%)"),
        ("Recall",   "Recall (%)"),
        ("Precision","Precision (%)"),
    ]

    fig6, axes6 = plt.subplots(2, 2, figsize=(16, 12))
    fig6.suptitle("Detection Metrics vs Gaussian Noise Level (No Denoising)",
                  fontsize=20, fontweight="bold")

    for ax, (metric, title) in zip(axes6.flat, _LINE_METRICS):
        if metric not in _df_noisy_base.columns:
            ax.set_visible(False)
            continue
        for model_name in _STD_MODELS:
            _df_m = _df_noisy_base[_df_noisy_base["model"] == model_name].sort_values("noise_sigma")
            if _df_m.empty:
                continue
            ax.plot(
                _df_m["noise_sigma"].values,
                _df_m[metric].values * 100,
                marker=_MODEL_MARKERS[model_name],
                linewidth=2.5, markersize=8,
                label=model_name,
                color=_MODEL_COLORS[model_name],
            )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("%", fontsize=14)
        ax.set_xticks(_SIGMAS)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    _fig6_path = os.path.join(PAPER_FIG_DIR, "Figure6_line_noisy_baseline.png")
    plt.savefig(_fig6_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  ✅ Figure 6 saved: {_fig6_path}")

    # ──────────────────────────────────────────────────────────────────────
    # FIGURE 7: HEATMAP — 4 metrics × model × noise level
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VẼ FIGURE 7: Heatmap - Noisy Baseline")
    print(SEP)

    _HM_METRICS = [
        ("mAP50",    "mAP@50 (%)"),
        ("mAP50-95", "mAP@50-95 (%)"),
        ("Precision","Precision (%)"),
        ("Recall",   "Recall (%)"),
    ]
    _HM_METRICS = [(m, t) for m, t in _HM_METRICS if m in _df_noisy_base.columns]

    fig7, axes7 = plt.subplots(2, 2, figsize=(18, 12))
    fig7.suptitle("Heatmap: Detection Metrics vs Gaussian Noise Level (No Denoising)",
                  fontsize=20, fontweight="bold")

    for ax, (metric, title) in zip(axes7.flat, _HM_METRICS):
        _pivot = _df_noisy_base.pivot_table(
            index="model", columns="noise_sigma", values=metric
        ) * 100
        _pivot = _pivot.reindex(index=_STD_MODELS, columns=sorted(_pivot.columns))
        _pivot.index = _pivot.index.astype(str)
        sns.heatmap(
            _pivot, annot=True, fmt=".1f",
            cmap="YlOrRd",
            linewidths=1, linecolor="white",
            annot_kws={"size": 13, "fontweight": "bold"},
            ax=ax,
            cbar_kws={"label": "Score (%)", "shrink": 0.8},
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=13)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("")

    plt.tight_layout()
    _fig7_path = os.path.join(PAPER_FIG_DIR, "Figure7_heatmap_noisy_baseline.png")
    plt.savefig(_fig7_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  ✅ Figure 7 saved: {_fig7_path}")

else:
    print("  ⚠️  Không tìm thấy results_checkpoint.csv → bỏ qua Figure 6 & 7")


# ──────────────────────────────────────────────────────────────────────
# FIGURE 9: LINE CHART — 4 metrics theo noise level (Gaussian Filter)
# FIGURE 10: HEATMAP — 4 metrics × model × noise level (Gaussian Filter)
# ──────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  VẼ FIGURE 9 & 10: Gaussian Filter Denoised Data")
print(SEP)

# Load Gaussian Filter denoised data
_gauss_paths = [
    os.path.join(RESULTS_DIR, "classification_metrics.csv"),
    os.path.join(OUTPUT_DIR, "results_summary", "classification_metrics.csv"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo", "classification_metrics.csv"),
]
_df_gauss_cls = None
for _p in _gauss_paths:
    if os.path.exists(_p):
        _tmp = pd.read_csv(_p)
        _tmp.columns = _tmp.columns.str.strip()
        if "denoise_method" in _tmp.columns:
            _gauss_only = _tmp[_tmp["denoise_method"] == "gaussian_filter"]
            if not _gauss_only.empty:
                _df_gauss_cls = _gauss_only.copy()
                print(f"  ✅ Gaussian Filter cls metrics: {_p}")
                break

# Load detection metrics from checkpoint
_df_gauss_det = None
if _df_ckpt is not None:
    _df_gauss_det = _df_ckpt[
        (_df_ckpt["denoise_method"] == "gaussian_filter") &
        (_df_ckpt["model"].isin(_STD_MODELS))
    ].copy()
    print(f"  ✅ Gaussian Filter det metrics: {len(_df_gauss_det)} rows from checkpoint")

# Merge detection + classification
if _df_gauss_det is not None:
    _df_gauss = _df_gauss_det.copy()
    if _df_gauss_cls is not None:
        _cls_cols = ["model", "noise_sigma", "Accuracy", "F1-Score", "Sensitivity", "Specificity"]
        _cls_cols = [c for c in _cls_cols if c in _df_gauss_cls.columns]
        _df_gauss = pd.merge(
            _df_gauss, _df_gauss_cls[_cls_cols],
            on=["model", "noise_sigma"], how="left"
        )
    _df_gauss = _df_gauss[_df_gauss["model"].isin(_STD_MODELS)]
    print(f"  ✅ Combined Gaussian Filter data: {len(_df_gauss)} rows")

    _SIGMAS = [0, 1, 5, 10, 20, 30]

    # ── FIGURE 9: Line chart ──
    _LINE_METRICS_G = [
        ("mAP50",    "mAP@50 (%)"),
        ("mAP50-95", "mAP@50-95 (%)"),
        ("Recall",   "Recall (%)"),
        ("Precision","Precision (%)"),
    ]
    _LINE_METRICS_G = [(m, t) for m, t in _LINE_METRICS_G if m in _df_gauss.columns]

    fig9, axes9 = plt.subplots(2, 2, figsize=(16, 12))
    fig9.suptitle("Detection Metrics vs Gaussian Noise Level (After Gaussian Filter Denoising)",
                  fontsize=20, fontweight="bold")

    for ax, (metric, title) in zip(axes9.flat, _LINE_METRICS_G):
        for model_name in _STD_MODELS:
            _df_m = _df_gauss[_df_gauss["model"] == model_name].sort_values("noise_sigma")
            if _df_m.empty:
                continue
            ax.plot(
                _df_m["noise_sigma"].values,
                _df_m[metric].values * 100,
                marker=_MODEL_MARKERS[model_name],
                linewidth=2.5, markersize=8,
                label=model_name,
                color=_MODEL_COLORS[model_name],
            )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("%", fontsize=14)
        ax.set_xticks(_SIGMAS)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    _fig9_path = os.path.join(PAPER_FIG_DIR, "Figure9_line_gaussian_filter.png")
    plt.savefig(_fig9_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  ✅ Figure 9 saved: {_fig9_path}")

    # ── FIGURE 10: Heatmap ──
    print(f"\n{SEP}")
    print("  VẼ FIGURE 10: Heatmap - Gaussian Filter")
    print(SEP)

    _HM_METRICS_G = [
        ("mAP50",    "mAP@50 (%)"),
        ("mAP50-95", "mAP@50-95 (%)"),
        ("Precision","Precision (%)"),
        ("Recall",   "Recall (%)"),
    ]
    _HM_METRICS_G = [(m, t) for m, t in _HM_METRICS_G if m in _df_gauss.columns]

    fig10, axes10 = plt.subplots(2, 2, figsize=(18, 12))
    fig10.suptitle("Heatmap: Detection Metrics After Gaussian Filter Denoising",
                   fontsize=20, fontweight="bold")

    for ax, (metric, title) in zip(axes10.flat, _HM_METRICS_G):
        _pivot = _df_gauss.pivot_table(
            index="model", columns="noise_sigma", values=metric
        ) * 100
        _pivot = _pivot.reindex(index=_STD_MODELS, columns=sorted(_pivot.columns))
        _pivot.index = _pivot.index.astype(str)
        sns.heatmap(
            _pivot, annot=True, fmt=".1f",
            cmap="YlGnBu",
            linewidths=1, linecolor="white",
            annot_kws={"size": 13, "fontweight": "bold"},
            ax=ax,
            cbar_kws={"label": "Score (%)", "shrink": 0.8},
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=13)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
        ax.set_xlabel("Noise Level (σ)", fontsize=14)
        ax.set_ylabel("")

    plt.tight_layout()
    _fig10_path = os.path.join(PAPER_FIG_DIR, "Figure10_heatmap_gaussian_filter.png")
    plt.savefig(_fig10_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  ✅ Figure 10 saved: {_fig10_path}")

else:
    print("  ⚠️  Gaussian Filter data not found → ignore Figures 9 & 10")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 12–15: COMPARE 6 DENOISING METHODS × 5 YOLO MODELS
# Source: results_checkpoint.csv (mAP50, mAP50-95, Precision, Recall)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  DRAW FIGURE 12–15: Comparing 6 Denoising Methods")
print(SEP)

_ALL_METHODS = ["noisy", "gaussian_filter", "bm3d", "autoencoder", "dncnn", "cae_pso"]
_METHOD_LABELS = {
    "noisy":           "Noisy (No Denoise)",
    "gaussian_filter": "Gaussian Filter",
    "bm3d":            "BM3D",
    "autoencoder":     "Autoencoder",
    "dncnn":           "DnCNN",
    "cae_pso":         "CAE+PSO",
}
_METHOD_COLORS = {
    "noisy":           "#888888",
    "gaussian_filter": "#E91E63",
    "bm3d":            "#9C27B0",
    "autoencoder":     "#2196F3",
    "dncnn":           "#4CAF50",
    "cae_pso":         "#FF9800",
}
_METHOD_MARKERS = {
    "noisy": "o", "gaussian_filter": "s", "bm3d": "^",
    "autoencoder": "D", "dncnn": "v", "cae_pso": "p",
}
_SIGMAS  = [0, 1, 5, 10, 20, 30]
_DENOISE_METHODS = [m for m in _ALL_METHODS if m != "noisy"]

if _df_ckpt is None:
    print("  ⚠️  Results_checkpoint.csv not found → skip Figure 12–15")
else:
    _df_all = _df_ckpt[_df_ckpt["model"].isin(_STD_MODELS)].copy()

    # ──────────────────────────────────────────────────────────────────
    # FIGURE 12: LINE CHART — mAP50 theo noise level, mỗi model 1 subplot
    # Mỗi line = 1 denoising method (5 methods + noisy baseline)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FIGURE 12: Line Chart - mAP50 per Model × Denoise Method")
    print(SEP)

    fig12, axes12 = plt.subplots(1, 5, figsize=(26, 5), sharey=False)
    fig12.suptitle("mAP@50: Effect of Denoising Methods on Each Model",
                   fontsize=13, fontweight="bold", y=1.02)

    _handles, _labels = [], []
    for idx_m, model_name in enumerate(_STD_MODELS):
        ax = axes12[idx_m]
        for method in _ALL_METHODS:
            _df_m = _df_all[
                (_df_all["model"] == model_name) &
                (_df_all["denoise_method"] == method)
            ].sort_values("noise_sigma")
            if _df_m.empty:
                continue
            lw = 2 if method == "noisy" else 1.8
            ls = "--" if method == "noisy" else "-"
            line, = ax.plot(
                _df_m["noise_sigma"].values,
                _df_m["mAP50"].values,
                marker=_METHOD_MARKERS[method],
                linewidth=lw, linestyle=ls, markersize=5,
                label=_METHOD_LABELS[method],
                color=_METHOD_COLORS[method],
            )
            if idx_m == 0:
                _handles.append(line)
                _labels.append(_METHOD_LABELS[method])

        _label = model_name.replace("yolov", "YOLOv").replace("yolo11", "YOLO11").replace("yolo12", "YOLO12")
        ax.set_title(_label, fontsize=11, fontweight="bold", pad=4)
        ax.set_xlabel("Noise Level (σ)", fontsize=10)
        if idx_m == 0:
            ax.set_ylabel("mAP@50", fontsize=10)
        ax.set_xticks(_SIGMAS)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig12.legend(_handles, _labels,
                 loc="center left",
                 bbox_to_anchor=(1.0, 0.5),
                 fontsize=9, framealpha=0.9,
                 title="Method", title_fontsize=9)

    plt.tight_layout()
    _fig12_path = os.path.join(PAPER_FIG_DIR, "Figure12_line_denoise_comparison.png")
    plt.savefig(_fig12_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  ✅ Figure 12 saved: {_fig12_path}")

    # ──────────────────────────────────────────────────────────────────
    # FIGURE 13: HEATMAP — mAP50 theo model × noise level
    # Each subplot = 1 denoising method
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FIGURE 13: Heatmap - mAP50 per Method × Model × Sigma")
    print(SEP)

    fig13, axes13 = plt.subplots(2, 3, figsize=(24, 14))


    for idx_meth, method in enumerate(_ALL_METHODS):
        ax = axes13.flat[idx_meth]
        _df_m = _df_all[_df_all["denoise_method"] == method]
        _pivot = _df_m.pivot_table(
            index="model", columns="noise_sigma", values="mAP50"
        ) * 100
        _pivot = _pivot.reindex(index=_STD_MODELS, columns=sorted(_pivot.columns))
        _pivot.index = _pivot.index.astype(str)

        _cmap = "YlOrRd" if method == "noisy" else "YlGnBu"
        sns.heatmap(
            _pivot, annot=True, fmt=".1f",
            cmap=_cmap,
            linewidths=0.8, linecolor="white",
            annot_kws={"size": 12, "fontweight": "bold"},
            ax=ax,
            cbar_kws={"label": "mAP@50 (%)", "shrink": 0.8},
            vmin=78, vmax=92,
        )
        ax.set_title(_METHOD_LABELS[method], fontsize=12, fontweight="bold", pad=4)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
        ax.set_xlabel("Noise Level (σ)", fontsize=12)
        ax.set_ylabel("")

    plt.tight_layout()
    _fig13_path = os.path.join(PAPER_FIG_DIR, "Figure13_heatmap_denoise_comparison.png")
    plt.savefig(_fig13_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  ✅ Figure 13 saved: {_fig13_path}")

    # ──────────────────────────────────────────────────────────────────
    # FIGURE 14: BAR CHART SO SÁNH — trung bình mAP50 & mAP50-95
    # Left:  grouped bar theo MODEL (avg across all methods & sigmas)
    # Right: grouped bar theo METHOD (avg across all models & sigmas)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FIGURE 14: Bar Chart - Average mAP50 & mAP50-95 Comparison")
    print(SEP)

    fig14, axes14 = plt.subplots(1, 2, figsize=(20, 7))

    # ── LEFT: Average mAP@50 & mAP@50-95 per MODEL (grouped 2 bars per model) ──
    _model_colors = [_MODEL_COLORS[m] for m in _STD_MODELS]
    _model_labels = [m.replace("yolov","YOLOv").replace("yolo11","YOLO11").replace("yolo12","YOLO12")
                     for m in _STD_MODELS]

    _avg50   = [_df_all[_df_all["model"]==m]["mAP50"].mean()    for m in _STD_MODELS]
    _avg95   = [_df_all[_df_all["model"]==m]["mAP50-95"].mean() for m in _STD_MODELS]

    _x   = np.arange(len(_STD_MODELS))
    _bw  = 0.35
    ax14l = axes14[0]

    # mAP@50 bars — darker (full color)
    bars50 = ax14l.bar(_x - _bw/2, _avg50, _bw,
                       color=_model_colors, label="mAP@50",
                       edgecolor="white", linewidth=0.5)
    # mAP@50-95 bars — lighter (same color, 50% alpha)
    bars95 = ax14l.bar(_x + _bw/2, _avg95, _bw,
                       color=_model_colors, alpha=0.5, label="mAP@50-95",
                       edgecolor="white", linewidth=0.5)

    # Value labels
    for bar in bars50:
        ax14l.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.002,
                   f"{bar.get_height():.3f}",
                   ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars95:
        ax14l.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.002,
                   f"{bar.get_height():.3f}",
                   ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax14l.set_xticks(_x)
    ax14l.set_xticklabels(_model_labels, fontsize=11)
    ax14l.set_ylabel("Score", fontsize=12)
    ax14l.set_title("Average mAP Across All Noise Levels & Methods",
                    fontsize=11, pad=8)
    ax14l.set_ylim(0.55, 0.95)
    ax14l.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax14l.grid(axis="y", alpha=0.3, linestyle="--")
    # Custom legend: 2 entries chỉ dùng màu xanh dương
    from matplotlib.patches import Patch
    ax14l.legend(handles=[
        Patch(facecolor="#2196F3", label="mAP@50"),
        Patch(facecolor="#2196F3", alpha=0.5, label="mAP@50-95"),
    ], fontsize=10, loc="upper right")

    # ── RIGHT: Average mAP@50 per MODEL × DENOISE METHOD (grouped 6 bars) ──
    ax14r = axes14[1]
    _n_methods = len(_ALL_METHODS)
    _total_bw  = 0.8
    _bw_each   = _total_bw / _n_methods
    _x2 = np.arange(len(_STD_MODELS))

    for idx_meth, method in enumerate(_ALL_METHODS):
        _vals = [_df_all[
                    (_df_all["model"]==m) &
                    (_df_all["denoise_method"]==method)
                 ]["mAP50"].mean()
                 for m in _STD_MODELS]
        offset = (idx_meth - _n_methods / 2 + 0.5) * _bw_each
        ax14r.bar(_x2 + offset, _vals, _bw_each,
                  color=_METHOD_COLORS[method],
                  label=_METHOD_LABELS[method],
                  edgecolor="white", linewidth=0.3)

    ax14r.set_xticks(_x2)
    ax14r.set_xticklabels(_model_labels, fontsize=11)
    ax14r.set_ylabel("mAP@50", fontsize=12)
    ax14r.set_title("Average mAP@50 by Model × Denoising Method",
                    fontsize=11, pad=8)
    ax14r.set_ylim(0.0, 1.0)
    ax14r.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax14r.grid(axis="y", alpha=0.3, linestyle="--")
    ax14r.legend(fontsize=9, loc="upper right",
                 title="", framealpha=0.9,
                 ncol=1)

    plt.tight_layout()
    _fig14_path = os.path.join(PAPER_FIG_DIR, "Figure14_bar_denoise_comparison.png")
    plt.savefig(_fig14_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  ✅ Figure 14 saved: {_fig14_path}")

    # ──────────────────────────────────────────────────────────────────
    # FIGURE 15: DELTA mAP50 — hiệu quả khử nhiễu so với noisy baseline
    # Left:  avg delta per method (bar chart)
    # Right: delta theo sigma per method (line chart)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FIGURE 15: Delta mAP50 - Denoising Effectiveness")
    print(SEP)

    # Tính delta = denoised - noisy (cùng model, cùng sigma)
    _noisy_base = _df_all[_df_all["denoise_method"] == "noisy"][
        ["model","noise_sigma","mAP50"]
    ].rename(columns={"mAP50": "mAP50_noisy"})

    _df_denoise = _df_all[_df_all["denoise_method"] != "noisy"].copy()
    _df_delta = pd.merge(_df_denoise, _noisy_base, on=["model","noise_sigma"], how="left")
    _df_delta["delta"] = _df_delta["mAP50"] - _df_delta["mAP50_noisy"]

    fig15, axes15 = plt.subplots(1, 2, figsize=(20, 8))


    # Left: average delta per method (bar, tất cả sigma)
    ax15l = axes15[0]
    _avg_deltas = []
    for method in _DENOISE_METHODS:
        avg = _df_delta[_df_delta["denoise_method"] == method]["delta"].mean()
        _avg_deltas.append(avg * 100)

    _bar_colors = [_METHOD_COLORS[m] for m in _DENOISE_METHODS]
    bars15 = ax15l.bar(range(len(_DENOISE_METHODS)), _avg_deltas,
                       color=_bar_colors, edgecolor="white", width=0.6)
    ax15l.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax15l.set_xticks(range(len(_DENOISE_METHODS)))
    ax15l.set_xticklabels([_METHOD_LABELS[m] for m in _DENOISE_METHODS],
                          rotation=0, ha="right", fontsize=11)
    ax15l.set_ylabel("Delta mAP@50 (pp)", fontsize=13)


    ax15l.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars15, _avg_deltas):
        va = "bottom" if val >= 0 else "top"
        offset = 0.05 if val >= 0 else -0.05
        ax15l.text(bar.get_x() + bar.get_width()/2, val + offset,
                   f"{val:.3f}", ha="center", va=va, fontsize=10, fontweight="bold")

    # Right: delta theo sigma per method (line chart, avg across models)
    ax15r = axes15[1]
    _sigmas_nozero = [1, 5, 10, 20, 30]
    for method in _DENOISE_METHODS:
        _delta_per_sigma = []
        for s in _sigmas_nozero:
            avg = _df_delta[
                (_df_delta["denoise_method"] == method) &
                (_df_delta["noise_sigma"] == s)
            ]["delta"].mean()
            _delta_per_sigma.append(avg * 100)
        ax15r.plot(_sigmas_nozero, _delta_per_sigma,
                   marker=_METHOD_MARKERS[method], linewidth=2.5, markersize=8,
                   label=_METHOD_LABELS[method], color=_METHOD_COLORS[method])

    ax15r.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax15r.set_xlabel("Noise Level (σ)", fontsize=13)
    ax15r.set_ylabel("Delta mAP@50 (pp)", fontsize=13)


    ax15r.set_xticks(_sigmas_nozero)
    ax15r.grid(True, alpha=0.3, linestyle="--")
    ax15r.legend(fontsize=10, loc="lower left")

    plt.tight_layout()
    _fig15_path = os.path.join(PAPER_FIG_DIR, "Figure15_delta_mAP_effectiveness.png")
    plt.savefig(_fig15_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  ✅ Figure 15 saved: {_fig15_path}")

    print(f"\n  ✅ Figure 12–15 hoàn tất")

# ──────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS FOR THE ARTICLE
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  📝 SUMMARY STATISTICS (for use in the descriptive paragraph of an article)")
print(f"{'=' * 70}")
for met in _all_metrics:
    vals     = dict(zip(df_orig['model'], df_orig[met].astype(float)))
    best_m   = max(vals, key=vals.get)
    worst_m  = min(vals, key=vals.get)
    print(f"  {met:12s}: BEST  = {best_m} ({vals[best_m]*100:.1f}%) | WORST = {worst_m} ({vals[worst_m]*100:.1f}%)")

print(f"\n  Range summary:")
for met in ['mAP50', 'mAP50-95']:
    vals_pct = [float(v) * 100 for v in df_orig[met]]
    print(f"  {met}: {min(vals_pct):.1f}% – {max(vals_pct):.1f}%")

_saved = sorted(glob.glob(os.path.join(PAPER_FIG_DIR, '*.png')))
print(f"\n  ✅ Tất cả {len(_saved)} figures đã lưu tại: {PAPER_FIG_DIR}")
for fp in _saved:
    print(f"    📄 {os.path.basename(fp)}")



# ============================================================================
# CELL 20: COMPARISON TABLES AND CHARTS
# ============================================================================
# Load from checkpoint if needed
ckpt_path = os.path.join(RESULTS_DIR, "results_checkpoint_full.csv")
if not all_results and os.path.exists(ckpt_path):
    df_all = pd.read_csv(ckpt_path)
    all_results = df_all.to_dict('records')
    print(f"✓ Loaded {len(all_results)} kết quả từ checkpoint")
else:
    df_all = pd.DataFrame(all_results)

print(f"\n{'═' * 80}")
print("  📋 BẢNG SO SÁNH: DENOISE METHODS × YOLO MODELS × NOISE LEVELS")
print(f"{'═' * 80}")

# ─── Pivot tables per metric ───
metrics_to_show = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'Composite']

for metric in metrics_to_show:
    if metric not in df_all.columns:
        continue
    print(f"\n{'─' * 80}")
    print(f"  📊 {metric}")
    print(f"{'─' * 80}")

    for model_name in sorted(df_all['model'].unique()):
        df_model = df_all[df_all['model'] == model_name]
        pivot = df_model.pivot_table(
            index='denoise_method', columns='noise_sigma',
            values=metric, aggfunc='first'
        )
        pivot = pivot.reindex(columns=sorted(pivot.columns))
        print(f"\n  {model_name}:")
        print((pivot * 100).round(2).to_string())

    # Save
    pivot_all = df_all.pivot_table(
        index=['model', 'denoise_method'], columns='noise_sigma',
        values=metric, aggfunc='first'
    ) * 100
    pivot_all.round(2).to_csv(os.path.join(RESULTS_DIR, f"pivot_{metric.replace('-','_')}.csv"))

# Save all results
df_all.to_csv(os.path.join(RESULTS_DIR, "all_results.csv"), index=False)
print(f"\n✅ Saved all results")


# ============================================================================
# CELL 21: DRAW A COMPARISON CHART
# ============================================================================
models = sorted(df_all['model'].unique())
methods = sorted(df_all['denoise_method'].unique())
sigmas = sorted(df_all['noise_sigma'].unique())

colors_method = plt.cm.Set2(np.linspace(0, 1, len(methods)))
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

metric_to_plot = 'mAP50-95'

# ─── 1. Line plots: Metric vs Noise cho TỪNG MODEL ───
fig, axes = plt.subplots(3, 3, figsize=(20, 12))
axes = axes.flatten()

for idx, model_name in enumerate(models):
    ax = axes[idx]
    df_model = df_all[df_all['model'] == model_name]

    for j, method in enumerate(methods):
        df_m = df_model[df_model['denoise_method'] == method].sort_values('noise_sigma')
        if df_m.empty:
            continue
        ax.plot(
            df_m['noise_sigma'], df_m[metric_to_plot] * 100,
            marker=markers[j % len(markers)], color=colors_method[j],
            linewidth=2, markersize=8, label=method, alpha=0.85,
        )
    ax.set_xlabel('Noise σ')
    ax.set_ylabel(f'{metric_to_plot} (%)')
    ax.set_title(model_name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sigmas)

# Hide unused subplot
if len(models) < len(axes):
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)

plt.suptitle(f'{metric_to_plot} vs Noise Level - Per Model (6 Denoise Methods)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'chart_denoise_per_model.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved per-model chart")

# ─── 2. Grouped bar: So sánh methods tại σ=20 (noise cao) ───
sigma_compare = 20
df_sigma = df_all[df_all['noise_sigma'] == sigma_compare]

if not df_sigma.empty:
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(models))
    n_methods = len(methods)
    width = 0.8 / n_methods

    for j, method in enumerate(methods):
        values = []
        for model in models:
            val = df_sigma[(df_sigma['model'] == model) & (df_sigma['denoise_method'] == method)][metric_to_plot].values
            values.append(val[0] * 100 if len(val) > 0 else 0)
        offset = (j - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors_method[j], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=6, fontweight='bold')

    ax.set_xlabel('YOLO Model', fontsize=13)
    ax.set_ylabel(f'{metric_to_plot} (%)', fontsize=13)
    ax.set_title(f'{metric_to_plot} tại σ={sigma_compare}: So sánh Denoise Methods', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'chart_bar_sigma{sigma_compare}.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved bar chart (σ={sigma_compare})")

# ─── 3. Heatmaps per model ───
for model_name in models:
    df_model = df_all[df_all['model'] == model_name]
    pivot = df_model.pivot_table(
        index='denoise_method', columns='noise_sigma',
        values=metric_to_plot, aggfunc='first'
    ) * 100
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                linewidths=0.5, cbar_kws={'label': f'{metric_to_plot} (%)'})
    ax.set_title(f'{model_name}: {metric_to_plot} (%) - Denoise Method × Noise', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'heatmap_{model_name}.png'), dpi=150, bbox_inches='tight')
    plt.show()

print("\n✅ All charts have been saved!")


# ============================================================================
# CELL 22: IMPROVEMENT TABLE - COMPARISON OF DENOISE VS. NOISY BASELINE
# ============================================================================
print(f"\n{'═' * 80}")
print("  📊 IMPROVEMENT: Denoise vs Noisy Baseline (percentage points)")
print(f"{'═' * 80}")

metric = 'mAP50-95'

for model_name in sorted(df_all['model'].unique()):
    df_model = df_all[df_all['model'] == model_name]
    pivot = df_model.pivot_table(
        index='denoise_method', columns='noise_sigma',
        values=metric, aggfunc='first'
    ) * 100
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    if 'noisy' in pivot.index:
        baseline = pivot.loc['noisy']
        improvement = pivot.subtract(baseline, axis=1)
        print(f"\n  {model_name} - {metric} improvement (pp vs noisy):")
        print(improvement.round(2).to_string())
        improvement.round(2).to_csv(
            os.path.join(RESULTS_DIR, f"improvement_{model_name}.csv")
        )

# ─── Summary table: Best method per model × sigma ───
print(f"\n{'─' * 80}")
print("  🏆 BEST DENOISE METHOD per Model × Noise Level")
print(f"{'─' * 80}")

for model_name in sorted(df_all['model'].unique()):
    df_model = df_all[df_all['model'] == model_name]
    print(f"\n  {model_name}:")
    for sigma in sorted(df_all['noise_sigma'].unique()):
        if sigma == 0:
            continue
        df_s = df_model[df_model['noise_sigma'] == sigma]
        if df_s.empty:
            continue
        best = df_s.loc[df_s[metric].idxmax()]
        print(f"    σ={sigma:2d}: {best['denoise_method']:>20s} → {metric}={best[metric]*100:.2f}%")


# ============================================================================
# CELL 23: SAVE ALL RESULTS TO GOOGLE DRIVE (VERSIONED)
# ============================================================================
import json

SAVE_DIR = LOCAL_SAVE_DIR  # Versions have been available since Cell 5
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 70)
print(f"  📤 SAVE THE RESULTS GOOGLE DRIVE")
print(f"  Version: {EXPERIMENT_VERSION}")
print(f"  Đích: {SAVE_DIR}")
print("=" * 70)

# ── 1. Results summary ──
dst_results = os.path.join(SAVE_DIR, "results_summary")
if os.path.exists(RESULTS_DIR):
    if os.path.exists(dst_results):
        shutil.rmtree(dst_results)
    shutil.copytree(RESULTS_DIR, dst_results)
    print(f"  ✓ Results summary → {dst_results}")

# ── 2. Denoise models (Skip if the previous version already has it.) ──
dst_models = os.path.join(SAVE_DIR, "denoise_models")
_prev_ver = os.path.join(_EXPERIMENT_BASE, f"ver_{EXPERIMENT_VERSION - 1}", "denoise_models") if EXPERIMENT_VERSION > 1 else ""
if os.path.exists(DENOISE_MODELS_DIR):
    if _prev_ver and os.path.exists(_prev_ver) and len(os.listdir(_prev_ver)) > 0:
        os.makedirs(dst_models, exist_ok=True)
        with open(os.path.join(dst_models, "_SEE_PREVIOUS_VERSION.txt"), 'w') as f:
            f.write(f"Denoise models remain unchanged.: {_prev_ver}\n")
        print(f"  ⏭ Denoise models → skip (giống ver_{EXPERIMENT_VERSION - 1})")
    else:
        if os.path.exists(dst_models):
            shutil.rmtree(dst_models)
        shutil.copytree(DENOISE_MODELS_DIR, dst_models)
        n_models = len(glob.glob(os.path.join(dst_models, "*.pt")))
        print(f"  ✓ Denoise models ({n_models} files)")

# ── 3. Best weights ──
best_weights_dir = os.path.join(SAVE_DIR, "best_weights")
os.makedirs(best_weights_dir, exist_ok=True)
n_weights = 0
for run_dir in sorted(glob.glob(os.path.join(TRAIN_OUTPUT_DIR, "*"))):
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        run_name = os.path.basename(run_dir)
        shutil.copy2(best_pt, os.path.join(best_weights_dir, f"{run_name}_best.pt"))
        n_weights += 1
print(f"  ✓ Best weights ({n_weights} files)")

# ── 4. Last weights ──
last_weights_dir = os.path.join(SAVE_DIR, "last_weights")
os.makedirs(last_weights_dir, exist_ok=True)
n_last = 0
for run_dir in sorted(glob.glob(os.path.join(TRAIN_OUTPUT_DIR, "*"))):
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    if os.path.exists(last_pt):
        run_name = os.path.basename(run_dir)
        shutil.copy2(last_pt, os.path.join(last_weights_dir, f"{run_name}_last.pt"))
        n_last += 1
print(f"  ✓ Last weights ({n_last} files)")

# ── 5. Training CSVs ──
train_csvs_dir = os.path.join(SAVE_DIR, "training_csvs")
os.makedirs(train_csvs_dir, exist_ok=True)
n_csvs = 0
for run_dir in sorted(glob.glob(os.path.join(TRAIN_OUTPUT_DIR, "*"))):
    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(results_csv):
        run_name = os.path.basename(run_dir)
        shutil.copy2(results_csv, os.path.join(train_csvs_dir, f"{run_name}_results.csv"))
        n_csvs += 1
print(f"  ✓ Training CSVs ({n_csvs} files)")

# ── 6. Training plots ──
train_plots_dir = os.path.join(SAVE_DIR, "training_plots")
os.makedirs(train_plots_dir, exist_ok=True)
n_plots = 0
for run_dir in sorted(glob.glob(os.path.join(TRAIN_OUTPUT_DIR, "*"))):
    for img_name in ['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png',
                      'F1_curve.png', 'P_curve.png', 'R_curve.png', 'PR_curve.png',
                      'val_batch0_pred.jpg', 'val_batch0_labels.jpg']:
        img_path = os.path.join(run_dir, img_name)
        if os.path.exists(img_path):
            run_name = os.path.basename(run_dir)
            shutil.copy2(img_path, os.path.join(train_plots_dir, f"{run_name}_{img_name}"))
            n_plots += 1
print(f"  ✓ Training plots ({n_plots} files)")

# ── 7. Notebook ──
notebook_src = '/content/YOLO_Denoise_Experiment_Karthy_COLAB.ipynb'
if os.path.exists(notebook_src):
    shutil.copy2(notebook_src, os.path.join(SAVE_DIR, "YOLO_Denoise_Experiment_Karthy_COLAB.ipynb"))
    print(f"  ✓ Notebook saved")

# ── 8. Experiment metadata ──
_meta = {
    'version': EXPERIMENT_VERSION,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models': list(YOLO_MODELS.keys()),
    'noise_levels': NOISE_LEVELS,
    'denoise_methods': DENOISE_METHODS,
    'total_results': len(all_results) if 'all_results' in dir() else 'N/A',
    'notes': f'Ver {EXPERIMENT_VERSION} - epochs=500, patience=50, classification metrics'
}
with open(os.path.join(SAVE_DIR, "experiment_meta.json"), 'w') as f:
    json.dump(_meta, f, indent=2, ensure_ascii=False)
print(f"  ✓ Metadata → experiment_meta.json")

# ── 9. Tổng kết ──
print(f"\n{'=' * 70}")
print(f"  ✅ ĐÃ LƯU VỀ GOOGLE DRIVE (Version {EXPERIMENT_VERSION})")
print(f"{'=' * 70}")
print(f"\n📁 {SAVE_DIR}")
for item in sorted(os.listdir(SAVE_DIR)):
    full = os.path.join(SAVE_DIR, item)
    if os.path.isdir(full):
        n = len([f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))])
        size_mb = sum(os.path.getsize(os.path.join(full, f)) for f in os.listdir(full)
                      if os.path.isfile(os.path.join(full, f))) / 1024**2
        print(f"  📁 {item:30s} {n:4d} files  ({size_mb:8.1f} MB)")
    else:
        size_mb = os.path.getsize(full) / 1024**2
        print(f"  📄 {item:30s}             ({size_mb:8.1f} MB)")

total_size = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, filenames in os.walk(SAVE_DIR)
                 for f in filenames) / 1024**3
print(f"\n  📊 Tổng dung lượng: {total_size:.2f} GB")

# ── 10. Danh sách versions ──
print(f"\n📋 Tất cả versions:")
for vd in sorted(os.listdir(_EXPERIMENT_BASE)):
    if not vd.startswith('ver_'):
        continue
    vp = os.path.join(_EXPERIMENT_BASE, vd)
    if os.path.isdir(vp):
        meta_f = os.path.join(vp, "experiment_meta.json")
        ts = ""
        if os.path.exists(meta_f):
            try:
                with open(meta_f) as f:
                    ts = json.load(f).get('timestamp', '')
            except:
                ts = "(meta corrupt)"
        vs = sum(os.path.getsize(os.path.join(dp, fn))
                 for dp, _, fns in os.walk(vp) for fn in fns) / 1024**3
        cur = " ← HIỆN TẠI" if vd == f"ver_{EXPERIMENT_VERSION}" else ""
        print(f"  📁 {vd:12s}  {ts:20s}  ({vs:.2f} GB){cur}")



# ============================================================================
# CELL 24: CHART COMPARE V2 - NOISY / DENOISED / ORIGIN
# ============================================================================
# Saved to: RESULTS_DIR/Chart_Compare_Ver_1/
# ============================================================================

import matplotlib
matplotlib.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CHART_DIR = os.path.join(RESULTS_DIR, "Chart_Compare_Ver_2")
os.makedirs(CHART_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA - Tim tat ca checkpoint files co the
# ══════════════════════════════════════════════════════════════════════
_search_dirs = [
    RESULTS_DIR,
    os.path.join(OUTPUT_DIR, "results_summary"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo"),
]
_search_names = [
    "results_checkpoint.csv",
    "results_checkpoint_full.csv",
]

_all_dfs = []
_found_files = []
for _dir in _search_dirs:
    for _name in _search_names:
        _path = os.path.join(_dir, _name)
        if os.path.exists(_path):
            _df = pd.read_csv(_path)
            _df.columns = _df.columns.str.strip()
            _all_dfs.append(_df)
            _found_files.append(f"{_path} ({len(_df)} rows)")

if _all_dfs:
    df_all = pd.concat(_all_dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['model', 'denoise_method', 'noise_sigma'], keep='last')
else:
    raise FileNotFoundError("No checkpoint CSV found!")

print(f"  Found files:")
for f in _found_files:
    print(f"    {f}")
print(f"  Combined: {len(df_all)} unique rows")

# Filter standard YOLO
_STD_MODELS = ['yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m']
df = df_all[df_all['model'].isin(_STD_MODELS)].copy()
print(f"  Standard YOLO: {len(df)} rows")

# Show available methods
_available_methods = df['denoise_method'].unique().tolist()
print(f"  Available methods: {_available_methods}")
print(f"  Models: {df['model'].unique().tolist()}")
print(f"  Sigmas: {sorted(df['noise_sigma'].unique().tolist())}")

_MODEL_COLORS = {
    'yolov8m': '#E91E63', 'yolov9m': '#9C27B0',
    'yolov10m': '#2196F3', 'yolo11m': '#4CAF50', 'yolo12m': '#FF9800',
}
_MODEL_MARKERS = {
    'yolov8m': 'o', 'yolov9m': 's', 'yolov10m': 'D', 'yolo11m': '^', 'yolo12m': 'v',
}
_METRICS = ['mAP50', 'mAP50-95', 'Precision', 'Recall']
_METRIC_TITLES = {
    'mAP50': 'mAP@50 (%)', 'mAP50-95': 'mAP@50-95 (%)',
    'Precision': 'Precision (%)', 'Recall': 'Recall (%)',
}
_METHOD_LABELS = {
    'noisy': 'Noisy (No Denoise)',
    'gaussian_filter': 'Gaussian Filter', 'bm3d': 'BM3D',
    'autoencoder': 'Autoencoder', 'dncnn': 'DnCNN', 'cae_pso': 'CAE+PSO',
}

# ══════════════════════════════════════════════════════════════════════
# 0. LINE CHART: ORIGINAL DATA (sigma=0, noisy = clean)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  0. Line Chart + Bar - Original Data")
print(f"{'=' * 70}")

df_origin = df[(df['noise_sigma'] == 0) & (df['denoise_method'] == 'noisy')].copy()

if len(df_origin) > 0:
    models = df_origin['model'].tolist()
    colors = [_MODEL_COLORS.get(m, '#888') for m in models]
    x = np.arange(len(models))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detection Metrics on Original Data (Clean, No Noise)',
                 fontsize=20, fontweight='bold')

    for ax, metric in zip(axes.flat, _METRICS):
        values = df_origin[metric].values * 100
        bars = ax.bar(x, values, width=0.6, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title(_METRIC_TITLES[metric], fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha='center', fontsize=12)
        ax.set_ylabel('%', fontsize=14)
        y_min = max(min(values) - 3, 0)
        ax.set_ylim(y_min, min(max(values) + 3, 100))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    _save = os.path.join(CHART_DIR, '00_bar_original_data.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

# ══════════════════════════════════════════════════════════════════════
# 1. LINE CHART: NOISY BASELINE across noise levels
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  1. Line Chart - Noisy Baseline")
print(f"{'=' * 70}")

df_noisy = df[df['denoise_method'] == 'noisy'].copy()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Detection Metrics vs Noise Level (No Denoising)',
             fontsize=20, fontweight='bold')

for ax, metric in zip(axes.flat, _METRICS):
    for model_name in _STD_MODELS:
        df_m = df_noisy[df_noisy['model'] == model_name].sort_values('noise_sigma')
        if len(df_m) == 0:
            continue
        ax.plot(df_m['noise_sigma'].values, df_m[metric].values * 100,
                marker=_MODEL_MARKERS[model_name], linewidth=2.5, markersize=8,
                label=model_name, color=_MODEL_COLORS[model_name])
    ax.set_title(_METRIC_TITLES[metric], fontsize=16, fontweight='bold')
    ax.set_xlabel('Noise Level (sigma)', fontsize=14)
    ax.set_ylabel('%', fontsize=14)
    ax.set_xticks(NOISE_LEVELS)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best')

plt.tight_layout()
_save = os.path.join(CHART_DIR, '01_line_noisy_baseline.png')
plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  Saved: {_save}")

# ══════════════════════════════════════════════════════════════════════
# 2. LINE CHART: AFTER DENOISING (each method = 1 figure)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  2. Line Chart - After Denoising")
print(f"{'=' * 70}")

_DENOISE_METHODS = [m for m in ['gaussian_filter', 'bm3d', 'autoencoder', 'dncnn', 'cae_pso']
                    if m in _available_methods]

for method in _DENOISE_METHODS:
    df_method = df[df['denoise_method'] == method].copy()
    if len(df_method) == 0:
        print(f"  SKIP {method}: no data")
        continue

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Detection Metrics After {_METHOD_LABELS[method]} Denoising',
                 fontsize=20, fontweight='bold')

    for ax, metric in zip(axes.flat, _METRICS):
        for model_name in _STD_MODELS:
            df_m = df_method[df_method['model'] == model_name].sort_values('noise_sigma')
            if len(df_m) == 0:
                continue
            ax.plot(df_m['noise_sigma'].values, df_m[metric].values * 100,
                    marker=_MODEL_MARKERS[model_name], linewidth=2.5, markersize=8,
                    label=model_name, color=_MODEL_COLORS[model_name])
        ax.set_title(_METRIC_TITLES[metric], fontsize=16, fontweight='bold')
        ax.set_xlabel('Noise Level (sigma)', fontsize=14)
        ax.set_ylabel('%', fontsize=14)
        ax.set_xticks(NOISE_LEVELS)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')

    plt.tight_layout()
    _save = os.path.join(CHART_DIR, f'02_line_denoise_{method}.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

# ══════════════════════════════════════════════════════════════════════
# 3. HEATMAP: ORIGINAL DATA (model x metrics)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  3. Heatmap - Original Data")
print(f"{'=' * 70}")

if len(df_origin) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Heatmap: All Metrics on Original Data (Clean)',
                 fontsize=20, fontweight='bold')

    hm_data = df_origin.set_index('model')[_METRICS].reindex(_STD_MODELS) * 100
    sns.heatmap(hm_data, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=1, linecolor='white',
                annot_kws={'size': 14, 'fontweight': 'bold'}, ax=ax,
                cbar_kws={'label': 'Score (%)', 'shrink': 0.8})
    ax.set_xticklabels([_METRIC_TITLES.get(m, m) for m in _METRICS], rotation=0, ha='center', fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)
    ax.set_ylabel('')

    plt.tight_layout()
    _save = os.path.join(CHART_DIR, '03_heatmap_original_data.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

# ══════════════════════════════════════════════════════════════════════
# 4. HEATMAP: NOISY DATA (model x sigma, per metric)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  4. Heatmap - Noisy Data")
print(f"{'=' * 70}")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Heatmap: Detection Metrics vs Noise Level (No Denoising)',
             fontsize=20, fontweight='bold')

for ax, metric in zip(axes.flat, _METRICS):
    pivot = df_noisy.pivot_table(index='model', columns='noise_sigma', values=metric) * 100
    pivot = pivot.reindex(index=_STD_MODELS, columns=sorted(pivot.columns))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=1, linecolor='white',
                annot_kws={'size': 13, 'fontweight': 'bold'}, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(_METRIC_TITLES[metric], fontsize=16, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
    ax.set_xlabel('Noise Level (sigma)', fontsize=14)
    ax.set_ylabel('')

plt.tight_layout()
_save = os.path.join(CHART_DIR, '04_heatmap_noisy_data.png')
plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  Saved: {_save}")

# ══════════════════════════════════════════════════════════════════════
# 5. HEATMAP: DENOISED DATA (each method = 1 figure)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  5. Heatmap - Denoised Data")
print(f"{'=' * 70}")

for method in _DENOISE_METHODS:
    df_method = df[df['denoise_method'] == method].copy()
    if len(df_method) == 0:
        print(f"  SKIP {method}: no data")
        continue

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Heatmap: Detection Metrics After {_METHOD_LABELS[method]} Denoising',
                 fontsize=20, fontweight='bold')

    for ax, metric in zip(axes.flat, _METRICS):
        pivot = df_method.pivot_table(index='model', columns='noise_sigma', values=metric) * 100
        pivot = pivot.reindex(index=_STD_MODELS, columns=sorted(pivot.columns))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=1, linecolor='white',
                    annot_kws={'size': 13, 'fontweight': 'bold'}, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(_METRIC_TITLES[metric], fontsize=16, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=13)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
        ax.set_xlabel('Noise Level (sigma)', fontsize=14)
        ax.set_ylabel('')

    plt.tight_layout()
    _save = os.path.join(CHART_DIR, f'05_heatmap_denoise_{method}.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

# ── Summary ──
_n_files = len(glob.glob(os.path.join(CHART_DIR, '*.png')))
print(f"\n{'=' * 70}")
print(f"  COMPLETE: {_n_files} charts saved to {CHART_DIR}")
print(f"{'=' * 70}")


# ============================================================================
# CELL 25: EXCEL SUMMARY - ALL METRICS (Original / Noise / Denoise)
# ============================================================================
# Merge results_checkpoint.csv (mAP50, mAP50-95, Precision, Recall)
# + classification_metrics.csv (Accuracy, F1, Sensitivity, Specificity)
# Output: 3 Excel files in Chart_Compare_Ver_1/
# ============================================================================

CHART_DIR = os.path.join(RESULTS_DIR, "Chart_Compare_Ver_2")
os.makedirs(CHART_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# LOAD + MERGE DATA
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  Loading data...")
print("=" * 70)

# 1. Detection metrics (mAP50, mAP50-95, Precision, Recall)
_det_paths = [
    os.path.join(RESULTS_DIR, "results_checkpoint.csv"),
    os.path.join(RESULTS_DIR, "results_checkpoint_full.csv"),
    os.path.join(OUTPUT_DIR, "results_summary", "results_checkpoint.csv"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo", "results_checkpoint.csv"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo", "results_checkpoint_full.csv"),
]

df_det = None
for p in _det_paths:
    if os.path.exists(p):
        df_det = pd.read_csv(p)
        df_det.columns = df_det.columns.str.strip()
        print(f"  Detection: {p} ({len(df_det)} rows)")
        break

# 2. Classification metrics (Accuracy, F1, Sensitivity, Specificity)
_cls_paths = [
    os.path.join(RESULTS_DIR, "classification_metrics.csv"),
    os.path.join(OUTPUT_DIR, "results_summary", "classification_metrics.csv"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo", "classification_metrics.csv"),
]

df_cls = None
for p in _cls_paths:
    if os.path.exists(p):
        df_cls = pd.read_csv(p)
        df_cls.columns = df_cls.columns.str.strip()
        print(f"  Classification: {p} ({len(df_cls)} rows)")
        break

# 3. Origin data metrics
_origin_paths = [
    os.path.join(RESULTS_DIR, "origin_data_metrics.csv"),
    os.path.join(OUTPUT_DIR, "results_summary", "origin_data_metrics.csv"),
    os.path.join(OUTPUT_DIR, "results_summary_yolo", "origin_data_metrics.csv"),
]

df_origin = None
for p in _origin_paths:
    if os.path.exists(p):
        df_origin = pd.read_csv(p)
        df_origin.columns = df_origin.columns.str.strip()
        print(f"  Origin: {p} ({len(df_origin)} rows)")
        break

# ── Merge detection + classification ──
if df_det is not None and df_cls is not None:
    # Chon cot can thiet tu classification
    _cls_cols = ['model', 'noise_sigma', 'denoise_method', 'Accuracy', 'F1-Score',
                 'Sensitivity', 'Specificity']
    _cls_available = [c for c in _cls_cols if c in df_cls.columns]
    df_cls_slim = df_cls[_cls_available].copy()

    # Merge
    df_merged = pd.merge(
        df_det, df_cls_slim,
        on=['model', 'noise_sigma', 'denoise_method'],
        how='left'
    )
    df_merged = df_merged.drop_duplicates(subset=['model', 'denoise_method', 'noise_sigma'], keep='last')
    print(f"  Merged: {len(df_merged)} rows (detection + classification)")
elif df_det is not None:
    df_merged = df_det.copy()
    print(f"  Using detection only: {len(df_merged)} rows (classification_metrics.csv not found)")
else:
    raise FileNotFoundError("No results_checkpoint.csv found!")

# Filter standard YOLO
_STD_MODELS = ['yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m']
df_merged = df_merged[df_merged['model'].isin(_STD_MODELS)].copy()

# Columns to export
_EXPORT_COLS = ['model', 'noise_sigma', 'denoise_method']
_METRIC_COLS = ['mAP50', 'mAP50-95', 'Precision', 'Recall',
                'Accuracy', 'F1-Score', 'Sensitivity', 'Specificity']
_EXTRA_COLS = ['TP', 'FP', 'FN', 'TN']

# Only include columns that exist
_all_cols = _EXPORT_COLS + [c for c in _METRIC_COLS if c in df_merged.columns] + \
            [c for c in _EXTRA_COLS if c in df_merged.columns]

print(f"  Available metrics: {[c for c in _METRIC_COLS if c in df_merged.columns]}")

# ══════════════════════════════════════════════════════════════════════
# EXCEL 1: ORIGINAL DATA (sigma=0, noisy)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  Excel 1: Original Data")
print(f"{'=' * 70}")

_xlsx1 = os.path.join(CHART_DIR, "01_Results_Original_Data.xlsx")

if df_origin is not None:
    with pd.ExcelWriter(_xlsx1, engine='openpyxl') as writer:
        df_origin.to_excel(writer, sheet_name='Original Data', index=False)
    print(f"  Saved: {_xlsx1}")
    print(df_origin.to_string(index=False))
else:
    # Fallback: lay tu merged
    df_orig_merged = df_merged[(df_merged['noise_sigma'] == 0) & (df_merged['denoise_method'] == 'noisy')].copy()
    _cols = [c for c in _all_cols if c in df_orig_merged.columns and c != 'denoise_method']
    df_orig_export = df_orig_merged[_cols].sort_values('model').reset_index(drop=True)

    with pd.ExcelWriter(_xlsx1, engine='openpyxl') as writer:
        df_orig_export.to_excel(writer, sheet_name='Original Data', index=False)
    print(f"  Saved: {_xlsx1}")
    print(df_orig_export.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# EXCEL 2: NOISY DATA (all sigma, denoise_method = 'noisy')
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  Excel 2: Noisy Data")
print(f"{'=' * 70}")

_xlsx2 = os.path.join(CHART_DIR, "02_Results_Noisy_Data.xlsx")
df_noisy = df_merged[df_merged['denoise_method'] == 'noisy'].copy()

_cols_noisy = ['model', 'noise_sigma'] + [c for c in _METRIC_COLS if c in df_noisy.columns] + \
              [c for c in _EXTRA_COLS if c in df_noisy.columns]

with pd.ExcelWriter(_xlsx2, engine='openpyxl') as writer:
    # Sheet 1: All data
    df_noisy_export = df_noisy[_cols_noisy].sort_values(['model', 'noise_sigma']).reset_index(drop=True)
    df_noisy_export.to_excel(writer, sheet_name='All Noisy', index=False)

    # Sheet per model
    for model in _STD_MODELS:
        df_m = df_noisy[df_noisy['model'] == model][_cols_noisy].sort_values('noise_sigma').reset_index(drop=True)
        if len(df_m) > 0:
            df_m.to_excel(writer, sheet_name=model, index=False)

    # Sheet per sigma
    for sigma in sorted(df_noisy['noise_sigma'].unique()):
        df_s = df_noisy[df_noisy['noise_sigma'] == sigma][_cols_noisy].sort_values('model').reset_index(drop=True)
        if len(df_s) > 0:
            df_s.to_excel(writer, sheet_name=f'sigma_{int(sigma)}', index=False)

print(f"  Saved: {_xlsx2} ({len(df_noisy_export)} rows)")

# ══════════════════════════════════════════════════════════════════════
# EXCEL 3: DENOISED DATA (all methods except 'noisy')
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  Excel 3: Denoised Data")
print(f"{'=' * 70}")

_xlsx3 = os.path.join(CHART_DIR, "03_Results_Denoised_Data.xlsx")
df_denoised = df_merged[df_merged['denoise_method'] != 'noisy'].copy()

_cols_denoise = ['model', 'noise_sigma', 'denoise_method'] + \
                [c for c in _METRIC_COLS if c in df_denoised.columns] + \
                [c for c in _EXTRA_COLS if c in df_denoised.columns]

_METHOD_LABELS = {
    'gaussian_filter': 'Gaussian Filter', 'bm3d': 'BM3D',
    'autoencoder': 'Autoencoder', 'dncnn': 'DnCNN', 'cae_pso': 'CAE+PSO',
}

with pd.ExcelWriter(_xlsx3, engine='openpyxl') as writer:
    # Sheet 1: All denoised data
    df_den_export = df_denoised[_cols_denoise].sort_values(
        ['denoise_method', 'model', 'noise_sigma']
    ).reset_index(drop=True)
    df_den_export.to_excel(writer, sheet_name='All Denoised', index=False)

    # Sheet per denoise method
    for method in sorted(df_denoised['denoise_method'].unique()):
        df_m = df_denoised[df_denoised['denoise_method'] == method][_cols_denoise].sort_values(
            ['model', 'noise_sigma']
        ).reset_index(drop=True)
        if len(df_m) > 0:
            sheet_name = _METHOD_LABELS.get(method, method)[:31]  # Excel max 31 chars
            df_m.to_excel(writer, sheet_name=sheet_name, index=False)

    # Sheet per model (all methods)
    for model in _STD_MODELS:
        df_m = df_denoised[df_denoised['model'] == model][_cols_denoise].sort_values(
            ['denoise_method', 'noise_sigma']
        ).reset_index(drop=True)
        if len(df_m) > 0:
            df_m.to_excel(writer, sheet_name=f'{model}_denoise', index=False)

print(f"  Saved: {_xlsx3} ({len(df_den_export)} rows)")

# ══════════════════════════════════════════════════════════════════════
# EXCEL 4: FULL COMPARISON (tat ca trong 1 file)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  Excel 4: Full Comparison (All in One)")
print(f"{'=' * 70}")

_xlsx4 = os.path.join(CHART_DIR, "04_Results_Full_Comparison.xlsx")

with pd.ExcelWriter(_xlsx4, engine='openpyxl') as writer:
    # Sheet: All data
    df_full = df_merged[_all_cols].sort_values(
        ['denoise_method', 'noise_sigma', 'model']
    ).reset_index(drop=True)
    df_full.to_excel(writer, sheet_name='All Results', index=False)

    # Sheet: Summary - avg per model across all conditions
    _metric_available = [c for c in _METRIC_COLS if c in df_merged.columns]
    df_summary = df_merged.groupby('model')[_metric_available].mean().reindex(_STD_MODELS)
    df_summary.to_excel(writer, sheet_name='Avg per Model')

    # Sheet: Summary - avg per method
    df_method_avg = df_merged.groupby('denoise_method')[_metric_available].mean()
    df_method_avg.to_excel(writer, sheet_name='Avg per Method')

    # Sheet: Summary - avg per sigma
    df_sigma_avg = df_merged.groupby('noise_sigma')[_metric_available].mean()
    df_sigma_avg.to_excel(writer, sheet_name='Avg per Sigma')

    # Sheet: Ranking - best model per metric per condition
    rankings = []
    for method in df_merged['denoise_method'].unique():
        for sigma in sorted(df_merged['noise_sigma'].unique()):
            df_cond = df_merged[(df_merged['denoise_method'] == method) & (df_merged['noise_sigma'] == sigma)]
            if len(df_cond) == 0:
                continue
            row = {'denoise_method': method, 'noise_sigma': sigma}
            for metric in _metric_available:
                if metric in df_cond.columns:
                    best_idx = df_cond[metric].idxmax()
                    row[f'best_{metric}'] = df_cond.loc[best_idx, 'model']
                    row[f'{metric}_value'] = round(df_cond[metric].max(), 4)
            rankings.append(row)
    if rankings:
        pd.DataFrame(rankings).to_excel(writer, sheet_name='Rankings', index=False)

print(f"  Saved: {_xlsx4} ({len(df_full)} rows)")

# ── Summary ──
print(f"\n{'=' * 70}")
print(f"  COMPLETE: 4 Excel files saved to {CHART_DIR}")
print(f"{'=' * 70}")
print(f"  01_Results_Original_Data.xlsx  - YOLO on clean data")
print(f"  02_Results_Noisy_Data.xlsx     - YOLO on noisy data (per model, per sigma)")
print(f"  03_Results_Denoised_Data.xlsx  - YOLO after denoising (per method, per model)")
print(f"  04_Results_Full_Comparison.xlsx - All results + rankings + averages")


# ============================================================================
# CELL 26: PREDICT BẰNG YOLO TRAINED ON DENOISED DATA
# ============================================================================
# Moi denoise method = 1 hinh tong hop
# Rows = 5 YOLO models, Cols = Original + sigma=1,5,10,20,30
# ============================================================================

import matplotlib
matplotlib.rcParams['font.size'] = 16
import matplotlib.pyplot as plt

# ── Tim TRAIN_OUTPUT_DIR ──
_possible_dirs = [
    os.path.join(OUTPUT_DIR, "training_runs"),
    '/content/training_runs_local',
]
TRAIN_OUTPUT_DIR = None
for _d in _possible_dirs:
    if os.path.exists(_d) and len(os.listdir(_d)) > 1:
        TRAIN_OUTPUT_DIR = _d
        break
if TRAIN_OUTPUT_DIR is None:
    TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training_runs")

# ── Tim sample image ──
with open(DATA_YAML_PATH, 'r') as f:
    _data_cfg = yaml.safe_load(f)
_dataset_root = _data_cfg.get('path', os.path.dirname(DATA_YAML_PATH))
_val_img_dir = os.path.join(_dataset_root, _data_cfg['val'])
_sample_candidates = sorted(glob.glob(os.path.join(_val_img_dir, '*')))

_sample_img_path = _sample_candidates[0]
for p in _sample_candidates:
    if 'image (1)' in os.path.basename(p):
        _sample_img_path = p
        break

_sample_img = cv2.imread(_sample_img_path)
print(f"  Sample image: {os.path.basename(_sample_img_path)} ({_sample_img.shape})")
print(f"  TRAIN_OUTPUT_DIR: {TRAIN_OUTPUT_DIR}")

# ── Reload denoise models ──
if not denoise_models:
    print("  Reload denoise models...")
    for sigma in NOISE_LEVELS:
        if sigma == 0:
            continue
        ae_path = os.path.join(DENOISE_MODELS_DIR, f"autoencoder_sigma{sigma}.pt")
        if os.path.exists(ae_path):
            m = DenoisingAutoencoder().to(DEVICE)
            m.load_state_dict(torch.load(ae_path, map_location=DEVICE)); m.eval()
            denoise_models[('autoencoder', sigma)] = m
        dn_path = os.path.join(DENOISE_MODELS_DIR, f"dncnn_sigma{sigma}.pt")
        if os.path.exists(dn_path):
            m = DnCNN(channels=3, num_layers=17, features=64).to(DEVICE)
            m.load_state_dict(torch.load(dn_path, map_location=DEVICE)); m.eval()
            denoise_models[('dncnn', sigma)] = m
        cae_path = os.path.join(DENOISE_MODELS_DIR, f"cae_pso_sigma{sigma}.pt")
        if os.path.exists(cae_path):
            ckpt = torch.load(cae_path, map_location=DEVICE)
            _p = _infer_cae_params_from_state_dict(ckpt)
            m = CAE(base_filters=_p['base_filters'], kernel_size=_p['kernel_size']).to(DEVICE)
            m.load_state_dict(ckpt); m.eval()
            denoise_models[('cae_pso', sigma)] = m
    print(f"  Loaded {len(denoise_models)} denoise models")

CHART_DIR = os.path.join(RESULTS_DIR, "Chart_Compare_Ver_1")
os.makedirs(CHART_DIR, exist_ok=True)

def _apply_denoise(img_orig, method, sigma):
    noisy = add_gaussian_noise(img_orig, sigma) if sigma > 0 else img_orig.copy()
    if method == 'gaussian_filter':
        return denoise_gaussian_filter(noisy, sigma=sigma)
    elif method == 'bm3d':
        return denoise_bm3d(noisy, sigma=sigma)
    elif method in ['autoencoder', 'dncnn', 'cae_pso']:
        if (method, sigma) in denoise_models:
            return denoise_with_model_fast(noisy, denoise_models[(method, sigma)])
    return noisy

def _predict_img(img_bgr, model_path):
    _tmp = '/tmp/_pred_denoise.jpg'
    cv2.imwrite(_tmp, img_bgr)
    _model = YOLO(model_path)
    results = _model.predict(source=_tmp, conf=0.25, iou=0.5, verbose=False, device=DEVICE_ID)
    result_img = results[0].plot()
    n_det = len(results[0].boxes)
    del _model; gc.collect(); torch.cuda.empty_cache()
    return result_img, n_det

_STD_MODELS = ['yolov8m', 'yolov9m', 'yolov10m', 'yolo11m', 'yolo12m']
_METHODS = ['gaussian_filter', 'bm3d', 'autoencoder', 'dncnn', 'cae_pso']
_METHOD_LABELS = {
    'gaussian_filter': 'Gaussian Filter', 'bm3d': 'BM3D',
    'autoencoder': 'Autoencoder', 'dncnn': 'DnCNN', 'cae_pso': 'CAE+PSO',
}
_sigmas = [s for s in NOISE_LEVELS if s > 0]

# ══════════════════════════════════════════════════════════════════════
# MOI METHOD = 1 HINH TONG HOP
# Rows = models, Cols = Original + sigma=1,5,10,20,30
# ══════════════════════════════════════════════════════════════════════

for method in _METHODS:
    method_label = _METHOD_LABELS[method]
    n_models = len(_STD_MODELS)
    n_cols = len(_sigmas) + 1  # +1 cho Original

    fig, axes = plt.subplots(n_models, n_cols, figsize=(n_cols * 4, n_models * 3.5))
    fig.suptitle(f'YOLO Prediction After {method_label} Denoising',
                 fontsize=20, fontweight='bold', y=1.02)

    for row_idx, model_name in enumerate(_STD_MODELS):
        # Cot 0: Original (predict anh sach bang model noisy_noise_0)
        ax = axes[row_idx, 0]
        bp_origin = os.path.join(TRAIN_OUTPUT_DIR, f"{model_name}_noisy_noise_0", "weights", "best.pt")
        if os.path.exists(bp_origin):
            result_img, n_det = _predict_img(_sample_img, bp_origin)
            ax.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            if row_idx == 0:
                ax.set_title(f'Original', fontsize=14, fontweight='bold')
        else:
            ax.imshow(cv2.cvtColor(_sample_img, cv2.COLOR_BGR2RGB))
            if row_idx == 0:
                ax.set_title('Original', fontsize=14, fontweight='bold')
        ax.axis('off')
        # Ghi ten model len goc trai anh
        ax.text(0.02, 0.98, model_name, transform=ax.transAxes,
                fontsize=13, fontweight='bold', color='white', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # Cot 1+: denoise + predict
        for col_idx, sigma in enumerate(_sigmas):
            ax = axes[row_idx, col_idx + 1]
            run_name = f"{model_name}_{method}_noise_{sigma}"
            best_pt = os.path.join(TRAIN_OUTPUT_DIR, run_name, "weights", "best.pt")

            if os.path.exists(best_pt):
                denoised_img = _apply_denoise(_sample_img, method, sigma)
                result_img, n_det = _predict_img(denoised_img, best_pt)
                ax.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                if row_idx == 0:
                    ax.set_title(f'sigma={sigma}', fontsize=14, fontweight='bold')
            else:
                ax.imshow(cv2.cvtColor(add_gaussian_noise(_sample_img, sigma), cv2.COLOR_BGR2RGB))
                if row_idx == 0:
                    ax.set_title(f'sigma={sigma}\n(no model)', fontsize=14)
            ax.axis('off')

    plt.tight_layout()
    _save = os.path.join(CHART_DIR, f'predict_denoise_{method}.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

print(f"\n{'=' * 70}")
print(f"  COMPLETE: 5 prediction images saved to {CHART_DIR}")
print(f"{'=' * 70}")


# ============================================================================
# CELL 27: NOISE + DENOISE + PREDICT IMAGES USING YOLO
# ============================================================================
# Phan 0: Comparing Gaussian levels
# Phan 1: Image after denoise (all methods x sigma)
# Phan 2: YOLO predicts noisy photos (one photo for each model)
# ============================================================================

import matplotlib
matplotlib.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
import numpy as np

# ── Tim TRAIN_OUTPUT_DIR ──
_possible_dirs = [
    os.path.join(OUTPUT_DIR, "training_runs"),
    '/content/training_runs_local',
]
TRAIN_OUTPUT_DIR = None
for _d in _possible_dirs:
    if os.path.exists(_d) and len(os.listdir(_d)) > 1:
        TRAIN_OUTPUT_DIR = _d
        break
if TRAIN_OUTPUT_DIR is None:
    TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "training_runs")

# ── Tim sample image ──
with open(DATA_YAML_PATH, 'r') as f:
    _data_cfg = yaml.safe_load(f)
_dataset_root = _data_cfg.get('path', os.path.dirname(DATA_YAML_PATH))
_val_img_dir = os.path.join(_dataset_root, _data_cfg['val'])
_sample_candidates = sorted(glob.glob(os.path.join(_val_img_dir, '*')))

_sample_img_path = _sample_candidates[0]
for p in _sample_candidates:
    if 'image (1)' in os.path.basename(p):
        _sample_img_path = p
        break

_sample_img = cv2.imread(_sample_img_path)
_sample_name = os.path.basename(_sample_img_path)
print(f"  Sample image: {_sample_name} ({_sample_img.shape})")

# ── Reload denoise models neu can ──
if not denoise_models:
    print("  Reload denoise models tu disk...")
    for sigma in NOISE_LEVELS:
        if sigma == 0:
            continue
        ae_path = os.path.join(DENOISE_MODELS_DIR, f"autoencoder_sigma{sigma}.pt")
        if os.path.exists(ae_path):
            ae_model = DenoisingAutoencoder().to(DEVICE)
            ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
            ae_model.eval()
            denoise_models[('autoencoder', sigma)] = ae_model
        dncnn_path = os.path.join(DENOISE_MODELS_DIR, f"dncnn_sigma{sigma}.pt")
        if os.path.exists(dncnn_path):
            dncnn_model = DnCNN(channels=3, num_layers=17, features=64).to(DEVICE)
            dncnn_model.load_state_dict(torch.load(dncnn_path, map_location=DEVICE))
            dncnn_model.eval()
            denoise_models[('dncnn', sigma)] = dncnn_model
        cae_path = os.path.join(DENOISE_MODELS_DIR, f"cae_pso_sigma{sigma}.pt")
        if os.path.exists(cae_path):
            cae_ckpt = torch.load(cae_path, map_location=DEVICE)
            _p = _infer_cae_params_from_state_dict(cae_ckpt)
            cae_model = CAE(base_filters=_p['base_filters'], kernel_size=_p['kernel_size']).to(DEVICE)
            cae_model.load_state_dict(cae_ckpt)
            cae_model.eval()
            denoise_models[('cae_pso', sigma)] = cae_model
    print(f"  Loaded {len(denoise_models)} denoise models")

def _apply_denoise(img_orig, method_key, sigma):
    if method_key == 'original':
        return img_orig.copy()
    elif method_key == 'noisy':
        return add_gaussian_noise(img_orig, sigma)
    elif method_key == 'gaussian_filter':
        return denoise_gaussian_filter(add_gaussian_noise(img_orig, sigma), sigma=sigma)
    elif method_key == 'bm3d':
        return denoise_bm3d(add_gaussian_noise(img_orig, sigma), sigma=sigma)
    elif method_key in ['autoencoder', 'dncnn', 'cae_pso']:
        noisy = add_gaussian_noise(img_orig, sigma)
        if (method_key, sigma) in denoise_models:
            return denoise_with_model_fast(noisy, denoise_models[(method_key, sigma)])
        return noisy
    return img_orig

# ══════════════════════════════════════════════════════════════════════
# PHAN 0: COMPARISON OF GAUSSIAN ITEMS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  PHAN 0: So sanh cac muc nhieu Gaussian")
print(f"{'=' * 70}")

fig, axes = plt.subplots(1, len(NOISE_LEVELS) + 1, figsize=((len(NOISE_LEVELS) + 1) * 4, 4.5))
fig.suptitle('So sanh cac muc nhieu Gaussian', fontsize=18, fontweight='bold')

axes[0].imshow(cv2.cvtColor(_sample_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Goc (sigma=0)', fontsize=14, fontweight='bold')
axes[0].axis('off')

for idx, sigma in enumerate(NOISE_LEVELS):
    ax = axes[idx + 1]
    if sigma == 0:
        display = _sample_img.copy()
    else:
        display = add_gaussian_noise(_sample_img, sigma)
    ax.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    ax.set_title(f'sigma={sigma}', fontsize=14)
    ax.axis('off')

plt.tight_layout()
_save = os.path.join(RESULTS_DIR, 'noise_levels_comparison.png')
plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  Saved: {_save}")

# ══════════════════════════════════════════════════════════════════════
# PHAN 1: IMAGE AFTER DENOISE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  PHAN 1: Anh sau denoise")
print(f"{'=' * 70}")

_methods_display = [
    ('original', 'Original'),
    ('noisy', 'Noisy'),
    ('gaussian_filter', 'Gaussian Filter'),
    ('bm3d', 'BM3D'),
    ('autoencoder', 'Autoencoder'),
    ('dncnn', 'DnCNN'),
    ('cae_pso', 'CAE+PSO'),
]

_sigmas_show = [s for s in NOISE_LEVELS if s > 0]

# Bang tong hop
fig, axes = plt.subplots(len(_sigmas_show), len(_methods_display),
                         figsize=(len(_methods_display) * 4, len(_sigmas_show) * 3.5))
fig.suptitle('Denoising Results Comparison', fontsize=20, fontweight='bold', y=1.02)

for row_idx, sigma in enumerate(_sigmas_show):
    for col_idx, (method_key, method_label) in enumerate(_methods_display):
        ax = axes[row_idx, col_idx]
        display_img = _apply_denoise(_sample_img, method_key, sigma)
        ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(method_label, fontsize=14, fontweight='bold')
        if col_idx == 0:
            ax.text(0.02, 0.98, f'sigma={sigma}', transform=ax.transAxes,
                    fontsize=13, fontweight='bold', color='white', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

plt.tight_layout()
_save = os.path.join(RESULTS_DIR, 'denoise_all_methods_all_sigma.png')
plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  Saved: {_save}")

# Moi method 1 hinh rieng
for method_key, method_label in _methods_display[2:]:
    fig, axes = plt.subplots(1, len(NOISE_LEVELS) + 1, figsize=((len(NOISE_LEVELS) + 1) * 3.5, 4))
    fig.suptitle(f'Denoise: {method_label}', fontsize=18, fontweight='bold')
    axes[0].imshow(cv2.cvtColor(_sample_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    for idx, sigma in enumerate(NOISE_LEVELS):
        ax = axes[idx + 1]
        if sigma == 0:
            display_img = _sample_img.copy()
        else:
            display_img = _apply_denoise(_sample_img, method_key, sigma)
        ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'sigma={sigma}', fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    _save = os.path.join(RESULTS_DIR, f'denoise_{method_key}_per_sigma.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

# So sanh tat ca methods tai sigma=30
_sigma_worst = max(NOISE_LEVELS)
fig, axes = plt.subplots(1, len(_methods_display), figsize=(len(_methods_display) * 3.5, 4))
fig.suptitle(f'All Denoise Methods at sigma={_sigma_worst}', fontsize=18, fontweight='bold')
for idx, (mk, ml) in enumerate(_methods_display):
    img = _apply_denoise(_sample_img, mk, _sigma_worst)
    axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(ml, fontsize=12)
    axes[idx].axis('off')
plt.tight_layout()
_save = os.path.join(RESULTS_DIR, f'denoise_comparison_sigma{_sigma_worst}.png')
plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  Saved: {_save}")

# ══════════════════════════════════════════════════════════════════════
# PHAN 2: YOLO PREDICT ON ANH NOISY (EACH MODEL HAS ITS OWN IMAGE)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  PHAN 2: YOLO Predict tren anh noisy")
print(f"{'=' * 70}")

def _predict_and_get(img_bgr, model_path):
    """Predict va tra ve anh co bbox + so detect."""
    _tmp = os.path.join('/tmp', '_predict_tmp.jpg')
    cv2.imwrite(_tmp, img_bgr)
    _model = YOLO(model_path)
    results = _model.predict(source=_tmp, conf=0.25, iou=0.5, verbose=False, device=DEVICE_ID)
    result_img = results[0].plot()
    n_det = len(results[0].boxes)
    del _model; gc.collect(); torch.cuda.empty_cache()
    return result_img, n_det

# Load models
_model_list = []
for model_name in YOLO_MODELS:
    run_name = f"{model_name}_noisy_noise_0"
    best_pt = os.path.join(TRAIN_OUTPUT_DIR, run_name, "weights", "best.pt")
    if os.path.exists(best_pt):
        _model_list.append((model_name, best_pt))
    else:
        print(f"  SKIP {model_name}: khong co best.pt")

_all_sigmas = [0] + [s for s in NOISE_LEVELS if s > 0]

if _model_list:
    # ── Moi model 1 hinh rieng: cac cot = sigma 0, 1, 5, 10, 20, 30 ──
    for model_name, best_pt in _model_list:
        fig, axes = plt.subplots(1, len(_all_sigmas), figsize=(len(_all_sigmas) * 4, 4.5))
        fig.suptitle(f'{model_name} - Predict on Noisy Images', fontsize=18, fontweight='bold')

        for col_idx, sigma in enumerate(_all_sigmas):
            ax = axes[col_idx]
            if sigma == 0:
                img = _sample_img.copy()
            else:
                img = add_gaussian_noise(_sample_img, sigma)

            result_img, n_det = _predict_and_get(img, best_pt)
            ax.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f'sigma={sigma}\n({n_det} det)', fontsize=13, fontweight='bold')
            ax.axis('off')

        # Ghi ten model len goc trai anh dau tien
        axes[0].text(0.02, 0.98, model_name, transform=axes[0].transAxes,
                     fontsize=13, fontweight='bold', color='yellow', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        plt.tight_layout()
        _save = os.path.join(RESULTS_DIR, f'predict_{model_name}_noisy.png')
        plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"  Saved: {_save}")

    # ── Tong hop: moi model 1 hang, cac cot = sigma ──
    n_models = len(_model_list)
    fig, axes = plt.subplots(n_models, len(_all_sigmas),
                             figsize=(len(_all_sigmas) * 4, n_models * 3.5))
    fig.suptitle('All YOLO Models - Predict on Noisy Images',
                 fontsize=20, fontweight='bold', y=1.02)
    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (model_name, best_pt) in enumerate(_model_list):
        for col_idx, sigma in enumerate(_all_sigmas):
            ax = axes[row_idx, col_idx]
            if sigma == 0:
                img = _sample_img.copy()
            else:
                img = add_gaussian_noise(_sample_img, sigma)

            result_img, n_det = _predict_and_get(img, best_pt)
            ax.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(f'sigma={sigma}' if sigma > 0 else 'Original',
                            fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.text(0.02, 0.98, model_name, transform=ax.transAxes,
                    fontsize=13, fontweight='bold', color='white', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.tight_layout()
    _save = os.path.join(RESULTS_DIR, 'predict_all_models_all_sigma.png')
    plt.savefig(_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"  Saved: {_save}")

else:
    print("  NO MODELS AVAILABLE best.pt!")

print(f"\n  COMPLETE!")


