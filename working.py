# === INTELLIGENT REMOTE SENSING ANALYST - CONSOLIDATED VERSION ===
# Merges the best features from multiple prototype scripts (1.py, 2.py, 3.py, main.py)
# into a single, robust, and enhanced implementation.
#
# Key Features:
# - Few-shot classification for novel classes.
# - Open-set recognition (OSR) for unknown objects.
# - Advanced model loading with CLIP, LoRA, and CoOp.
# - Flexible data loading (local, Hugging Face, synthetic fallback).
# - Enhanced inference with multi-scale tiling and data augmentation.
# - Comprehensive PDF reporting with embedded plots and qualitative examples.

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
from collections import defaultdict, Counter
import time
import warnings
import pandas as pd
from fpdf import FPDF
import io
import base64
import math
import huggingface_hub
import sys
from tqdm import tqdm
import shutil
import tempfile
import xml.etree.ElementTree as ET
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# === SECTION: CONFIGURATION ===
class SystemConfig:
    """Consolidated system configuration, combining the best features from multiple prototype scripts."""
    def __init__(self):
        # Dataset configuration
        # NOTE: Update this path if your data is stored elsewhere.
        self.DATASET_PATH = "./data/RSVG-pytorch/DIOR_RSVG"
        self.HF_DATASET_NAME = "danielz01/DIOR-RSVG"
        self.HF_TOKEN = os.getenv("HF_TOKEN", "") # Replace with your token if needed

        # Operation mode
        self.DEMO_MODE = False  # Set to True for demo mode with limited functionality

        # Force local data loading
        self.USE_REAL_DATA_ONLY = True  # Changed to True to force local data loading

        # DIOR-RSVG Dataset Characteristics
        self.TOTAL_IMAGES = 17402
        self.TOTAL_QUERY_PAIRS = 38320
        self.IMAGE_SIZE = 800  # Original DIOR image size
        self.SPATIAL_RESOLUTION_RANGE = (0.5, 30.0)  # meters/pixel
        self.AVG_QUERY_LENGTH = 7.47
        
        # Class configuration (20 classes from DIOR)
        self.ALL_CLASSES = [
            'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
            'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
            'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
            'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
        ]

        # Adaptive class splits for few-shot and open-set tasks
        self.BASE_CLASSES = self.ALL_CLASSES[:10]   # For calibration
        self.NOVEL_CLASSES = self.ALL_CLASSES[10:18] # For few-shot evaluation
        self.UNKNOWN_CLASSES = self.ALL_CLASSES[18:] # For open-set evaluation

        # Create mappings
        self.class_to_label = {c: i for i, c in enumerate(self.ALL_CLASSES)}
        self.label_to_class = {i: c for c, i in self.class_to_label.items()}

        # Episode configuration
        self.N_WAY = 5
        self.K_SHOT = 5
        self.Q_QUERY = 5

        # Full performance parameters
        self.CALIB_EPISODES = 50
        self.EVAL_EPISODES = 50

        # Model configuration
        self.MODEL_NAME = "openai/clip-vit-base-patch16"
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.IMAGE_SIZE = 224  # CLIP input size
        self.BATCH_SIZE = 32
        self.SEED = 42

        # LoRA configuration
        self.LORA_ENABLED = True
        self.LORA_R = 8
        self.LORA_ALPHA = 16
        self.LORA_DROPOUT = 0.1

        # CoOp configuration
        self.COOP_ENABLED = True
        self.COOP_TOKENS = 10

        # Inference configuration - Updated based on DIOR characteristics
        self.TILE_SIZES = [224, 336, 448]  # Multiple scales for different resolutions
        self.TILE_OVERLAP = [0.25, 0.5] # Reduced options for speed

        # Open-Set Recognition (OSR) configuration
        self.OSR_CALIBRATION_METHOD: str = 'advanced' # 'f1_max' or 'advanced'
        self.OSR_USE_PERCENTILE: bool = True
        self.OSR_PERCENTILE: float = 5.0
        self.OSR_USE_GMM: bool = True

        # Enhanced Prompt Templates based on DIOR-RSVG patterns
        self.PROMPT_TEMPLATES: tuple = (
            # Phrase Template patterns
            "a {} in aerial view",
            "a {} in satellite imagery",
            "a {} in remote sensing image",
            # Location-based patterns
            "a {} in the {} part of the image",
            "a {} near the landmark",
            "a {} next to the feature",
            # Attribute-based patterns
            "a {} {} {} in the scene",
            "a {} shaped {} from above",
            # Relationship-based patterns
            "a {} {} another {}",
            "a {} surrounded by {}"
        )

        # DIOR-RSVG Attribute Sets
        self.ATTRIBUTE_SETS = {
            'colors': ['white', 'gray', 'dark', 'light', 'metallic'],
            'sizes': ['small', 'medium', 'large', 'tiny', 'huge'],
            'geometry': ['rectangular', 'circular', 'linear', 'curved', 'irregular'],
            'locations': ['center', 'edge', 'corner', 'border'],
            'directions': ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest'],
            'relations': ['near', 'beside', 'between', 'among', 'parallel to', 'perpendicular to'],
            'contexts': ['buildings', 'vegetation', 'roads', 'water bodies', 'open space']
        }

        # Adaptation configuration
        self.ADAPTATION_STEPS = 30

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not callable(v)}

CONFIG = SystemConfig()

# === SECTION: ENVIRONMENT SETUP ===
def setup_environment(config):
    """Handles package installation, HF login, and seed setting."""
    print("--- Setting up Environment ---")
    try:
        # Install packages quietly
        install_commands = [
            "pip install -q torch torchvision timm transformers datasets scikit-learn peft fpdf2 tqdm torchgeo huggingface_hub matplotlib seaborn"
        ]
        for cmd in install_commands:
            # os.system is simple and effective for Colab
            os.system(cmd)

        # Login to HF if token provided
        if config.HF_TOKEN:
            try:
                huggingface_hub.login(token=config.HF_TOKEN, add_to_git_credential=False)
                print("✅ Hugging Face login successful")
            except Exception as e:
                print(f"⚠️ HF login failed: {e}. Public datasets should still work.")

        # Set random seeds for reproducibility
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


        print(f"✅ Environment setup complete. Using device: {config.DEVICE}")
        return config.DEVICE

    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        raise

# === SECTION: DATA LOADER ===
class EnhancedDiorDataset:
    """Robust dataset loader with local, Hugging Face, and synthetic data strategies."""
    def __init__(self, config):
        self.config = config
        print("--- Loading Enhanced DIOR Dataset ---")

        # Load data with fallback strategy
        self.dataset = self._load_with_fallback()

        # Create class splits
        self.base_classes, self.novel_classes, self.unknown_classes = self._create_class_splits()

        # Filter data by splits
        self.base_data = self._filter_by_classes(self.dataset, self.base_classes)
        self.novel_data = self._filter_by_classes(self.dataset, self.novel_classes)
        self.unknown_data = self._filter_by_classes(self.dataset, self.unknown_classes, label_as_unknown=True)

        print(f"📊 Dataset loaded: Base={len(self.base_data)}, Novel={len(self.novel_data)}, Unknown={len(self.unknown_data)}")

    def _load_with_fallback(self):
        """Try multiple loading strategies: local path, Hugging Face, synthetic."""
        # Strategy 1: Local path (if it exists and is a directory)
        if os.path.isdir(self.config.DATASET_PATH):
            try:
                print("Attempting to load data from local path...")
                return self._load_local()
            except Exception as e:
                print(f"⚠️ Local loading failed: {e}")

        # Strategy 2: Hugging Face
        try:
            print("Attempting to load data from Hugging Face...")
            return self._load_huggingface()
        except Exception as e:
            print(f"⚠️ Hugging Face loading failed: {e}")
            print("Trying alternative Hugging Face dataset...")
            try:
                # Try alternative dataset
                self.config.HF_DATASET_NAME = "danielz01/DIOR-RSVG"
                return self._load_huggingface()
            except Exception as e2:
                print(f"⚠️ Alternative Hugging Face loading failed: {e2}")

        # Strategy 3: Synthetic data (if real data is not required)
        if not self.config.USE_REAL_DATA_ONLY:
            print("🔄 Creating synthetic data as a last resort...")
            return self._create_synthetic_data()
        else:
            print("\n❌ All data loading strategies failed. Please try one of the following:")
            print("1. Download the DIOR dataset manually from: https://drive.google.com/drive/folders/1hTqtYsC6B-m4ED2ewx5oKuYZV13EoJp_")
            print("2. Set USE_REAL_DATA_ONLY=False in the config to use synthetic data")
            print("3. Check your Hugging Face token and internet connection")
            raise Exception("All data loading strategies failed and synthetic data is disabled. Check DATASET_PATH and HF_TOKEN.")

    def _load_local(self):
        """Load from a local directory structured like DIOR."""
        import glob
        data = []
        image_dir = os.path.join(self.config.DATASET_PATH, 'JPEGImages')
        annotation_dir = os.path.join(self.config.DATASET_PATH, 'Annotations')
        
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"JPEGImages directory not found in {self.config.DATASET_PATH}")
        if not os.path.isdir(annotation_dir):
            raise FileNotFoundError(f"Annotations directory not found in {self.config.DATASET_PATH}")

        # Create a mapping of image IDs to their annotations
        image_annotations = {}
        for xml_file in glob.glob(os.path.join(annotation_dir, '*.xml')):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get image ID from filename
                image_id = os.path.splitext(os.path.basename(xml_file))[0]
                
                # Get image size
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                
                # Get all object annotations
                objects = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.config.ALL_CLASSES:
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        # Get description if available
                        desc = obj.find('description')
                        description = desc.text if desc is not None else ""
                        
                        objects.append({
                            'label': name,
                            'bbox': [xmin, ymin, xmax, ymax],
                            'description': description
                        })
                
                if objects:
                    image_annotations[image_id] = {
                        'size': (width, height),
                        'objects': objects
                    }
                    
            except Exception as e:
                print(f"Warning: Could not parse annotation file {xml_file}: {e}")
                continue

        # Process each image
        for img_path in glob.glob(os.path.join(image_dir, '*.jpg')):
            try:
                img = Image.open(img_path).convert('RGB')
                image_id = os.path.splitext(os.path.basename(img_path))[0]
                
                # Get annotations for this image
                if image_id in image_annotations:
                    annotations = image_annotations[image_id]
                    for obj in annotations['objects']:
                        # Crop the object from the image
                        xmin, ymin, xmax, ymax = obj['bbox']
                        cropped_img = img.crop((xmin, ymin, xmax, ymax))
                        
                        # Add to dataset with metadata
                        data.append({
                            'image': cropped_img,
                            'label': obj['label'],
                            'bbox': obj['bbox'],
                            'description': obj['description'],
                            'image_size': annotations['size'],
                            'relative_position': {
                                'center_x': (xmin + xmax) / (2 * width),
                                'center_y': (ymin + ymax) / (2 * height),
                                'width': (xmax - xmin) / width,
                                'height': (ymax - ymin) / height
                            }
                        })
                else:
                    # If no annotation found, skip this image
                    continue
                    
            except Exception as e:
                print(f"Warning: Could not process image {img_path}: {e}")
                continue

        if not data:
            raise FileNotFoundError("No valid images with labels found in local path.")

        print(f"✅ Loaded {len(data)} samples from local path")
        return data

    def _load_huggingface(self):
        """Load from Hugging Face Hub with enhanced DIOR-RSVG support."""
        from datasets import load_dataset
        import time

        # Try different split options based on available data
        splits_to_try = ["train", "validation", "test"]
        dataset = None
        max_retries = 3
        retry_delay = 5  # seconds

        # Create a temporary directory for fresh downloads
        temp_dir = tempfile.mkdtemp()
        try:
            for retry in range(max_retries):
                try:
                    for split in splits_to_try:
                        try:
                            print(f"Attempting to load '{split}' split...")
                            # Force fresh download by using cache_dir
                            dataset = load_dataset(
                                self.config.HF_DATASET_NAME,
                                split=split,
                                token=self.config.HF_TOKEN,
                                cache_dir=temp_dir,
                                download_mode="force_redownload"
                            )
                            print(f"✅ Successfully loaded '{split}' split with {len(dataset)} samples")
                            break
                        except Exception as e:
                            print(f"Failed to load '{split}' split: {e}")
                            continue

                    if dataset is None:
                        # Try loading without specifying split
                        print("Trying to load full dataset...")
                        dataset = load_dataset(
                            self.config.HF_DATASET_NAME,
                            token=self.config.HF_TOKEN,
                            cache_dir=temp_dir,
                            download_mode="force_redownload"
                        )
                        # Use the first available split
                        available_splits = list(dataset.keys())
                        print(f"Available splits: {available_splits}")
                        dataset = dataset[available_splits[0]]
                        print(f"Using split '{available_splits[0]}' with {len(dataset)} samples")

                    data = []
                    max_samples = self.config.TOTAL_IMAGES  # Use actual dataset size

                    for item in tqdm(dataset, total=max_samples, desc="Processing DIOR-RSVG dataset"):
                        if len(data) >= max_samples:
                            break
                        try:
                            # Extract all objects and their descriptions from the image
                            image = item['image']
                            objects = item['objects']
                            
                            # Process each object in the image
                            for idx, (bbox, category, caption) in enumerate(zip(
                                objects['bbox'], 
                                objects['categories'], 
                                objects.get('descriptions', [''] * len(objects['bbox']))
                            )):
                                if category in self.config.ALL_CLASSES:
                                    # Validate and normalize bbox coordinates
                                    x1, y1, x2, y2 = bbox
                                    if x1 >= x2 or y1 >= y2:
                                        continue
                                        
                                    # Filter out too small or too large boxes
                                    box_area = (x2 - x1) * (y2 - y1)
                                    image_area = image.width * image.height
                                    if box_area / image_area < 0.0002 or box_area / image_area > 0.99:
                                        continue

                                    # Crop the object
                                    try:
                                        cropped_image = image.crop((x1, y1, x2, y2))
                                        if cropped_image.width > 10 and cropped_image.height > 10:
                                            # Store enhanced metadata
                                            data.append({
                                                'image': cropped_image,
                                                'label': category,
                                                'bbox': bbox,
                                                'caption': caption,
                                                'image_size': (image.width, image.height),
                                                'relative_position': {
                                                    'center_x': (x1 + x2) / (2 * image.width),
                                                    'center_y': (y1 + y2) / (2 * image.height),
                                                    'width': (x2 - x1) / image.width,
                                                    'height': (y2 - y1) / image.height
                                                }
                                            })
                                    except Exception:
                                        continue

                        except Exception:
                            continue  # Skip malformed entries

                    if not data:
                        raise ConnectionError("Could not extract any valid data from Hugging Face.")

                    print(f"✅ Loaded {len(data)} object instances from DIOR-RSVG")
                    
                    # Additional dataset statistics
                    class_counts = Counter(item['label'] for item in data)
                    print("\nClass distribution:")
                    for cls, count in class_counts.most_common():
                        print(f"  {cls}: {count} samples")
                    
                    return data

                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"Attempt {retry + 1} failed: {e}")
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise Exception(f"Failed to load dataset after {max_retries} attempts: {e}")

            raise Exception("Failed to load dataset from Hugging Face after all retries")
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory: {e}")

    def _create_synthetic_data(self):
        """Create synthetic data for demo purposes."""
        data = []
        for i in range(500):
            img = Image.new('RGB', (224, 224), color=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
            draw = ImageDraw.Draw(img)
            label = random.choice(self.config.ALL_CLASSES)
            draw.text((10, 100), label, fill=(255, 255, 255))
            data.append({'image': img, 'label': label})
        print(f"✅ Created {len(data)} synthetic samples")
        return data

    def _create_class_splits(self):
        """Create balanced class splits based on available data."""
        available_labels = list(set(item['label'] for item in self.dataset if item['label'] != 'unknown'))

        # Use predefined splits, filtering for what's available
        base = [c for c in self.config.BASE_CLASSES if c in available_labels]
        novel = [c for c in self.config.NOVEL_CLASSES if c in available_labels]
        unknown = [c for c in self.config.UNKNOWN_CLASSES if c in available_labels]

        # If data is limited, dynamically create splits
        if len(base) < 5 or len(novel) < 5 or len(unknown) < 2:
            print("⚠️ Limited data availability. Dynamically creating class splits.")
            random.shuffle(available_labels)
            base = available_labels[:10]
            novel = available_labels[10:18]
            unknown = available_labels[18:]
            if not unknown: # Ensure unknown set is not empty
                unknown = [novel.pop()]

        return base, novel, unknown

    def _filter_by_classes(self, data, target_classes, label_as_unknown=False):
        """Filter dataset by a list of target classes."""
        filtered = []
        for item in data:
            if item['label'] in target_classes:
                new_item = item.copy()
                if label_as_unknown:
                    new_item['label'] = 'unknown'
                filtered.append(new_item)
        return filtered

    def get_episode(self, split='novel'):
        """Generate a few-shot episode."""
        if split == 'novel':
            data_pool = self.novel_data
            class_pool = self.novel_classes
        else: # base
            data_pool = self.base_data
            class_pool = self.base_classes

        # Sample N_WAY classes for the episode
        episode_classes = random.sample(class_pool, min(self.config.N_WAY, len(class_pool)))
        support_set, query_set = [], []
        support_labels, query_labels = [], []

        for class_name in episode_classes:
            class_samples = [item for item in data_pool if item['label'] == class_name]
            if len(class_samples) < self.config.K_SHOT + self.config.Q_QUERY:
                if not class_samples: continue # Skip class if no samples
                # Repeat samples if not enough available
                class_samples *= math.ceil((self.config.K_SHOT + self.config.Q_QUERY) / len(class_samples))

            random.shuffle(class_samples)
            support_samples = class_samples[:self.config.K_SHOT]
            query_samples = class_samples[self.config.K_SHOT : self.config.K_SHOT + self.config.Q_QUERY]

            for item in support_samples:
                support_set.append(item['image'])
                support_labels.append(self.config.class_to_label[class_name])
            for item in query_samples:
                query_set.append(item['image'])
                query_labels.append(self.config.class_to_label[class_name])

        return support_set, support_labels, query_set, query_labels, episode_classes

    def get_unknown_samples(self, num_samples):
        """Get a specified number of samples from the unknown classes."""
        if not self.unknown_data: return []
        unknown_samples = random.sample(self.unknown_data, min(num_samples, len(self.unknown_data)))
        return [item['image'] for item in unknown_samples]

# === SECTION: MODEL ===
class LearnablePrompts(nn.Module):
    """CoOp learnable prompts implementation."""
    def __init__(self, n_tokens=10, token_dim=512, class_names=[], processor=None, device='cpu'):
        super().__init__()
        self.learnable_tokens = nn.Parameter(torch.randn(n_tokens, token_dim))
        self.n_tokens = n_tokens
        self.class_names = class_names
        self.processor = processor
        self.device = device
        self.token_dim = token_dim

    def forward(self):
        # This is a simplified forward for inference; a real implementation would
        # embed text and combine it with learnable_tokens.
        prompts = [f"aerial image of a {c.replace('-', ' ')}" for c in self.class_names]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        # In a real CoOp model, you would manipulate the token embeddings here.
        # For this prototype, we just return the standard text features.
        return inputs

def load_enhanced_clip_model(config, device):
    """Load CLIP with LoRA and CoOp enhancements."""
    print("--- Loading Enhanced CLIP Model ---")
    from transformers import CLIPModel, CLIPProcessor
    from peft import get_peft_model, LoraConfig

    model = CLIPModel.from_pretrained(config.MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)

    if config.LORA_ENABLED:
        try:
            lora_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            print("✅ LoRA applied successfully")
            model.print_trainable_parameters()
        except Exception as e:
            print(f"⚠️ LoRA setup failed: {e}. Continuing without LoRA.")

    coop_module = None
    if config.COOP_ENABLED:
        # A full CoOp implementation requires training. Here we initialize it
        # but will use standard text prompts for this inference-only script.
        coop_module = LearnablePrompts(
            n_tokens=config.COOP_TOKENS,
            token_dim=model.config.text_config.hidden_size,
            processor=processor,
            device=device
        ).to(device)
        print("✅ CoOp module initialized (inference mode)")

    model.eval()
    return model, processor, coop_module

# === SECTION: ADAPTATION ===
def run_adaptation(model, processor, dataset, config):
    """Runs a meta-learning based adaptation phase on the base classes."""
    print("\n--- Running Meta-Learning Adaptation on Base Classes ---")
    model.train()
    
    # Ensure LoRA parameters are trainable
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    # Get all trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    # Use a simplified transform for faster adaptation
    adaptation_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create progress bar with explicit range
    steps = range(config.ADAPTATION_STEPS)
    pbar = tqdm(steps, desc="Meta-Learning Adaptation")
    
    for _ in pbar:
        # Get an episode (support and query sets)
        support_set, support_labels, query_set, query_labels, episode_classes = dataset.get_episode(split='base')
        if not support_set or not query_set:
            pbar.update(1)
            continue

        # Map labels to 0..N-1 for CrossEntropyLoss
        unique_class_labels = sorted(list(set(support_labels)))
        label_map = {lbl: i for i, lbl in enumerate(unique_class_labels)}
        mapped_query_labels = torch.tensor([label_map[l] for l in query_labels], device=config.DEVICE)

        # Process support images to create prototypes
        with torch.no_grad():  # Prototypes are targets, no gradients needed
            support_tensors = torch.stack([adaptation_transform(img) for img in support_set]).to(config.DEVICE)
            support_features = model.get_image_features(pixel_values=support_tensors)
            support_features = F.normalize(support_features, dim=-1)

            # Create prototypes by averaging features per class
            prototypes = []
            for class_label in unique_class_labels:
                class_mask = torch.tensor([l == class_label for l in support_labels], device=config.DEVICE)
                if class_mask.any():
                    class_proto = support_features[class_mask].mean(dim=0)
                    prototypes.append(class_proto)
            prototypes = torch.stack(prototypes)
            prototypes = F.normalize(prototypes, dim=-1)

        # Process query images with gradients enabled
        query_tensors = torch.stack([adaptation_transform(img) for img in query_set]).to(config.DEVICE)
        query_features = model.get_image_features(pixel_values=query_tensors)
        query_features = F.normalize(query_features, dim=-1)

        # Calculate logits (similarities to prototypes)
        logits = torch.matmul(query_features, prototypes.T)

        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, mapped_query_labels)

        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    pbar.close()
    print("✅ Meta-learning adaptation complete.")
    model.eval()

# === SECTION: ENHANCED INFERENCE ===
class EnhancedInference:
    """Handles prototype creation and multi-scale inference."""
    def __init__(self, model, processor, coop_module, config, device):
        self.model = model
        self.processor = processor
        self.coop_module = coop_module
        self.config = config
        self.device = device

    def create_prototypes(self, support_set, support_labels, episode_classes):
        """Create class prototypes from the support set with enhanced DIOR-RSVG features."""
        # Process images with augmentations and metadata
        augmented_features = []
        for item in support_set:
            # Get image and metadata
            image = item['image'] if isinstance(item, dict) else item
            metadata = item if isinstance(item, dict) else {}
            
            # Basic augmentations: original + rotation
            aug_images = [image]
            for angle in [90, 180, 270]:
                aug_images.append(image.rotate(angle))

            # Process all augmentations in a single batch
            inputs = self.processor(images=aug_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(pixel_values=inputs.pixel_values)
                features = F.normalize(features, dim=-1)
                # Average features across augmentations
                avg_features = features.mean(dim=0, keepdim=True)
                augmented_features.append(avg_features)

        image_features = torch.cat(augmented_features, dim=0)

        # Create image prototypes with proper normalization
        n_classes = len(episode_classes)
        image_prototypes = torch.zeros(n_classes, image_features.size(1), device=self.device)
        class_counts = torch.zeros(n_classes, device=self.device)
        class_to_episode_idx = {name: i for i, name in enumerate(episode_classes)}

        for i, label in enumerate(support_labels):
            class_name = self.config.label_to_class[label]
            if class_name in episode_classes:
                idx = class_to_episode_idx[class_name]
                image_prototypes[idx] += image_features[i]
                class_counts[idx] += 1

        # Normalize prototypes with epsilon to avoid division by zero
        image_prototypes = F.normalize(image_prototypes / (class_counts.unsqueeze(1) + 1e-6), dim=-1)

        # Create text prototypes using enhanced prompt templates
        with torch.no_grad():
            class_prototypes_list = []
            for class_name in episode_classes:
                # Generate prompts using templates and attributes
                prompts = []
                
                # Basic templates - always include at least one basic prompt
                basic_prompt = f"aerial image of a {class_name.replace('-', ' ')}"
                prompts.append(basic_prompt)
                
                # Add additional prompts from templates
                for template in self.config.PROMPT_TEMPLATES:
                    try:
                        if '{' not in template:
                            # Simple template
                            prompts.append(template.format(class_name.replace('-', ' ')))
                        else:
                            # Template with attributes
                            # Add location-based prompts
                            if '{direction}' in template:
                                for direction in self.config.ATTRIBUTE_SETS['directions']:
                                    prompt = template.format(
                                        class_name.replace('-', ' '),
                                        direction=direction
                                    )
                                    prompts.append(prompt)
                            
                            # Add attribute-based prompts
                            elif '{size}' in template and '{color}' in template:
                                for size in self.config.ATTRIBUTE_SETS['sizes']:
                                    for color in self.config.ATTRIBUTE_SETS['colors']:
                                        prompt = template.format(
                                            class_name.replace('-', ' '),
                                            size=size,
                                            color=color,
                                            geometry=random.choice(self.config.ATTRIBUTE_SETS['geometry'])
                                        )
                                        prompts.append(prompt)
                            
                            # Add relationship-based prompts
                            elif '{relation}' in template:
                                for relation in self.config.ATTRIBUTE_SETS['relations']:
                                    other_class = random.choice([c for c in episode_classes if c != class_name])
                                    prompt = template.format(
                                        class_name.replace('-', ' '),
                                        relation=relation,
                                        object=other_class.replace('-', ' ')
                                    )
                                    prompts.append(prompt)
                            
                            # Add context-based prompts
                            elif '{context}' in template:
                                for context in self.config.ATTRIBUTE_SETS['contexts']:
                                    prompt = template.format(
                                        class_name.replace('-', ' '),
                                        context=context
                                    )
                                    prompts.append(prompt)
                                    
                    except (KeyError, IndexError) as e:
                        # If any template fails, just continue with the basic prompt
                        continue

                # Ensure we have at least one valid prompt
                if not prompts:
                    prompts = [basic_prompt]

                # Process all prompts for this class in a single batch
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                # Average the features across all prompts with proper normalization
                class_prototype = F.normalize(text_features.mean(dim=0), p=2, dim=-1)
                class_prototypes_list.append(class_prototype)
            
            text_prototypes = torch.stack(class_prototypes_list)
            text_prototypes = F.normalize(text_prototypes, dim=-1)  # Final normalization

        return image_prototypes, text_prototypes

    def multi_scale_inference(self, query_images):
        """Run inference using a highly efficient batch-processed, multi-scale sliding window."""
        all_tiles = []
        img_indices = []  # To map tiles back to their original image
        metadata_list = []  # Store metadata for each tile

        # First, collect all tiles from all images
        for i, item in enumerate(query_images):
            image = item['image'] if isinstance(item, dict) else item
            metadata = item if isinstance(item, dict) else {}
            
            # Adapt tile sizes based on image resolution if available
            if 'image_size' in metadata:
                orig_w, orig_h = metadata['image_size']
                adaptive_tile_sizes = [
                    min(ts, min(orig_w, orig_h) // 2) 
                    for ts in self.config.TILE_SIZES
                ]
            else:
                adaptive_tile_sizes = self.config.TILE_SIZES

            for tile_size in adaptive_tile_sizes:
                for overlap in self.config.TILE_OVERLAP:
                    tiles = self._sliding_window_tiles(image, tile_size, overlap)
                    if tiles:
                        all_tiles.extend(tiles)
                        img_indices.extend([i] * len(tiles))
                        # Store metadata for each tile
                        if 'relative_position' in metadata:
                            metadata_list.extend([metadata['relative_position']] * len(tiles))
                        else:
                            metadata_list.extend([None] * len(tiles))

        if not all_tiles:  # Fallback for images that produced no tiles
            inputs = self.processor(images=[img['image'] if isinstance(img, dict) else img for img in query_images], 
                                  return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(pixel_values=inputs.pixel_values)
                return F.normalize(features, dim=-1)

        # Process all tiles in batches
        all_tile_features = []
        for i in range(0, len(all_tiles), self.config.BATCH_SIZE):
            batch_tiles = all_tiles[i:i + self.config.BATCH_SIZE]
            inputs = self.processor(images=batch_tiles, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(pixel_values=inputs.pixel_values)
                features = F.normalize(features, dim=-1)
                all_tile_features.append(features)

        all_tile_features = torch.cat(all_tile_features, dim=0)

        # Apply position-based weighting if metadata is available
        if any(m is not None for m in metadata_list):
            weights = torch.ones(len(all_tile_features), device=self.device)
            for i, metadata in enumerate(metadata_list):
                if metadata is not None:
                    pos = metadata
                    center_weight = 1.0 - (
                        abs(pos['center_x'] - 0.5) + 
                        abs(pos['center_y'] - 0.5)
                    ) / 2.0
                    weights[i] = center_weight
            all_tile_features = all_tile_features * weights.unsqueeze(1)

        # Aggregate features for each original image using max-pooling
        num_images = len(query_images)
        output_dim = all_tile_features.size(1)
        aggregated_features = torch.zeros(num_images, output_dim, device=self.device)

        img_indices_tensor = torch.tensor(img_indices, device=self.device)
        for i in range(num_images):
            mask = (img_indices_tensor == i)
            if mask.any():
                # Max pooling over all tiles belonging to image `i`
                aggregated_features[i], _ = torch.max(all_tile_features[mask], dim=0)

        return F.normalize(aggregated_features, dim=-1)

    def _sliding_window_tiles(self, image, tile_size, overlap):
        """Generate tiles from an image using a sliding window."""
        w, h = image.size
        if tile_size > min(w, h): return [image.resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))]

        tiles = []
        step = int(tile_size * (1 - overlap))
        for y in range(0, h - tile_size + 1, step):
            for x in range(0, w - tile_size + 1, step):
                tiles.append(image.crop((x, y, x + tile_size, y + tile_size)))
        return tiles if tiles else [image.resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))]

    def predict(self, query_images, image_prototypes, text_prototypes):
        """Predict class for query images using combined image and text similarities."""
        with torch.no_grad():
            # Get features for all query images in one batch
            query_features = self.multi_scale_inference(query_images)

            # Compute similarities with both image and text prototypes
            img_similarities = torch.matmul(query_features, image_prototypes.T)
            text_similarities = torch.matmul(query_features, text_prototypes.T)

            # Combine similarities (simple average)
            combined_similarities = (img_similarities + text_similarities) / 2

            # Get predictions and confidences for all images at once
            max_scores, predictions = torch.max(combined_similarities, dim=1)
            confidences = max_scores.tolist()

        return predictions.tolist(), confidences

# === SECTION: OSR CALIBRATION ===
def calibrate_osr_threshold(inference_engine, dataset, device, config):
    """Calibrates the OSR threshold on the base classes."""
    print("--- Calibrating OSR Threshold ---")
    all_known_scores = []
    all_scores = []
    all_is_known = []
    
    # In calibration, "unknowns" are samples from the novel classes
    novel_data_pool = dataset.novel_data

    for _ in tqdm(range(config.CALIB_EPISODES), desc="OSR Calibration"):
        support_set, support_labels, query_set, query_labels, episode_classes = dataset.get_episode(split='base')
        if not support_set: continue

        # Add "unknowns" (from novel set) to the query set
        num_unknown_samples = len(query_set)
        if novel_data_pool:
            unknown_samples_data = random.sample(novel_data_pool, min(num_unknown_samples, len(novel_data_pool)))
            unknown_images = [item['image'] for item in unknown_samples_data]
            
            original_query_len = len(query_set)
            query_set.extend(unknown_images)
            is_known_labels = [1] * original_query_len + [0] * len(unknown_images)
        else:
            is_known_labels = [1] * len(query_set)

        img_prototypes, text_prototypes = inference_engine.create_prototypes(support_set, support_labels, episode_classes)
        _, confidences = inference_engine.predict(query_set, img_prototypes, text_prototypes)
        
        all_scores.extend(confidences)
        all_is_known.extend(is_known_labels)

    if not all_scores:
        print("⚠️ No calibration scores collected, using default threshold 0.5")
        return 0.5

    all_scores = np.array(all_scores)
    all_is_known = np.array(all_is_known)
    
    best_threshold = 0.0
    best_method = "default"
    best_score = -1.0
    
    # Method 1: Traditional F1 maximization
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 100)
    f1_scores = [f1_score(all_is_known, all_scores >= t, zero_division=0) for t in thresholds]
    
    if np.max(f1_scores) > best_score:
        best_score = np.max(f1_scores)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_method = "f1_max"
        print(f"F1-max threshold: {best_threshold:.4f} (F1: {best_score:.4f})")

    # Method 2: Percentile-based thresholding
    if config.OSR_USE_PERCENTILE:
        known_scores = all_scores[all_is_known == 1]
        if len(known_scores) > 0:
            percentile_threshold = np.percentile(known_scores, config.OSR_PERCENTILE)
            percentile_f1 = f1_score(all_is_known, all_scores >= percentile_threshold, zero_division=0)
            print(f"Percentile-{config.OSR_PERCENTILE} threshold: {percentile_threshold:.4f} (F1: {percentile_f1:.4f})")
            if percentile_f1 > best_score:
                best_score = percentile_f1
                best_threshold = percentile_threshold
                best_method = f"percentile_{config.OSR_PERCENTILE}"

    # Method 3: Gaussian Mixture Model-based thresholding
    if config.OSR_USE_GMM:
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=2, random_state=config.SEED).fit(all_scores.reshape(-1, 1))
            # Find the intersection of the two Gaussians
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            idx_low, idx_high = np.argmin(means), np.argmax(means)
            # A simple intersection point calculation
            gmm_threshold = (means[idx_high] * stds[idx_low] + means[idx_low] * stds[idx_high]) / (stds[idx_high] + stds[idx_low])

            gmm_f1 = f1_score(all_is_known, all_scores >= gmm_threshold, zero_division=0)
            print(f"GMM-based threshold: {gmm_threshold:.4f} (F1: {gmm_f1:.4f})")
            if gmm_f1 > best_score:
                best_score = gmm_f1
                best_threshold = gmm_threshold
                best_method = "gmm"
        except Exception as e:
            print(f"⚠️ GMM-based thresholding failed: {e}")

    print(f"✅ Best OSR method: '{best_method}', calibrated threshold: {best_threshold:.4f} (Score: {best_score:.4f})")
    return best_threshold

# === SECTION: EVALUATION ===
def evaluate_performance(inference_engine, dataset, osr_threshold, config):
    """Evaluates the model on the novel classes against unknown classes."""
    print("--- Evaluating Performance ---")
    all_predictions, all_labels, all_confidences = [], [], []
    from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score

    qualitative_results = [] # For visual examples in the report
    few_shot_accuracies = [] # Track accuracy for each episode

    with torch.no_grad(): # Evaluation is pure inference
        for episode_idx in tqdm(range(config.EVAL_EPISODES), desc="Evaluation"):
            support_set, support_labels, query_set, query_labels, episode_classes = dataset.get_episode(split='novel')
            if not support_set: continue

            # Add unknown samples to the query set for OSR evaluation
            unknown_images = dataset.get_unknown_samples(config.Q_QUERY * config.N_WAY)
            query_set.extend(unknown_images)
            query_labels.extend([-1] * len(unknown_images)) # -1 for unknown

            img_prototypes, text_prototypes = inference_engine.create_prototypes(support_set, support_labels, episode_classes)
            predictions, confidences = inference_engine.predict(query_set, img_prototypes, text_prototypes)

            # Apply OSR threshold and map predictions to global labels
            osr_predictions = []
            for pred_idx, conf in zip(predictions, confidences):
                if conf < osr_threshold:
                    osr_predictions.append(-1) # Predicted as unknown
                else:
                    class_name = episode_classes[pred_idx]
                    osr_predictions.append(config.class_to_label[class_name])

            # Calculate few-shot accuracy for this episode (only on known classes)
            known_mask = [label != -1 for label in query_labels]
            if any(known_mask):
                episode_accuracy = accuracy_score(
                    [query_labels[i] for i in range(len(query_labels)) if known_mask[i]],
                    [osr_predictions[i] for i in range(len(osr_predictions)) if known_mask[i]]
                )
                few_shot_accuracies.append(episode_accuracy)

            all_predictions.extend(osr_predictions)
            all_labels.extend(query_labels)
            all_confidences.extend(confidences)

            # Collect qualitative results for report (first few samples)
            if episode_idx == 0:
                for i in range(min(len(query_set), 8)):
                    qualitative_results.append({
                        'image': query_set[i],
                        'true_label': query_labels[i],
                        'pred_label': osr_predictions[i],
                        'confidence': confidences[i],
                    })

    # Compute overall metrics
    metrics = {}

    # Calculate few-shot accuracy statistics
    if few_shot_accuracies:
        metrics['few_shot_accuracy_mean'] = np.mean(few_shot_accuracies)
        metrics['few_shot_accuracy_std'] = np.std(few_shot_accuracies)
    else:
        metrics['few_shot_accuracy_mean'] = 0.0
        metrics['few_shot_accuracy_std'] = 0.0

    # --- Use scikit-learn for robust metric calculation ---
    # Separate known and unknown predictions
    known_mask = np.array(all_labels) != -1
    unknown_mask = ~known_mask

    y_true_known = np.array(all_labels)[known_mask]
    y_pred_known = np.array(all_predictions)[known_mask]
    
    # Get class names for reporting
    class_names = [config.label_to_class[label] for label in sorted(set(y_true_known))]
    class_names.append('unknown')  # Add unknown class
    
    # Create label mapping for sklearn
    label_map = {label: i for i, label in enumerate(sorted(set(y_true_known)))}
    label_map[-1] = len(label_map)  # Map unknown to last index
    
    # Convert labels to sklearn format
    y_true_sklearn = np.array([label_map[label] for label in all_labels])
    y_pred_sklearn = np.array([label_map[label] for label in all_predictions])
    
    # Get detailed classification report
    report = classification_report(
        y_true_sklearn, 
        y_pred_sklearn,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True
    )
    
    # Store metrics
    metrics['classification_report'] = report
    metrics['accuracy'] = report['accuracy']
    
    # Calculate AUROC for open-set detection
    binary_labels = [0 if label == -1 else 1 for label in all_labels]
    try:
        fpr, tpr, _ = roc_curve(binary_labels, all_confidences)
        metrics['auroc'] = auc(fpr, tpr)
    except Exception:
        metrics['auroc'] = 0.5
        fpr, tpr = None, None

    # Calculate UDF1 Score (Harmonic mean of known F1 and unknown accuracy)
    known_f1 = report['macro avg']['f1-score']  # Use macro F1 for known classes
    unknown_acc = report['unknown']['precision']  # Use precision as accuracy for unknown class
    
    metrics['udf1_score'] = (2 * known_f1 * unknown_acc) / (known_f1 + unknown_acc) if (known_f1 + unknown_acc) > 0 else 0.0

    # Store raw data for visualization
    metrics['raw_labels'] = all_labels
    metrics['raw_predictions'] = all_predictions

    # Print detailed results
    print("\n✅ Evaluation completed:")
    print(f"\nFew-shot Accuracy: {metrics['few_shot_accuracy_mean']:.3f} ± {metrics['few_shot_accuracy_std']:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_true_sklearn, 
        y_pred_sklearn,
        labels=range(len(class_names)),
        target_names=class_names
    ))
    print(f"\nAUROC: {metrics['auroc']:.4f}")
    print(f"UDF1 Score: {metrics['udf1_score']:.4f}")

    return metrics, (fpr, tpr), qualitative_results

# === SECTION: REPORT GENERATION ===
class PDF(FPDF):
    """Custom PDF class with header, footer, and helpers."""
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Intelligent Remote Sensing Analyst Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, data):
        self.set_font('Helvetica', '', 9)
        for k, v in data.items():
            self.multi_cell(self.w - self.l_margin - self.r_margin, 5, f"- {k}: {v}", align='L')
        self.ln()

    def add_plot(self, title, plot_path, width=160):
        if not os.path.exists(plot_path): return
        self.add_page()
        self.chapter_title(title)
        self.image(plot_path, x=None, y=None, w=width)
        self.ln(10)

def generate_enhanced_report(config, metrics, roc_data, qualitative_results, start_time):
    """Generates a comprehensive PDF report with plots and images."""
    print("--- Generating Enhanced Report ---")
    report_filename = "intelligent_analyst_report.pdf"
    plot_paths = {}

    # --- Generate Plots ---
    # Confusion Matrix
    try:
        all_labels = metrics['raw_labels']
        all_preds = metrics['raw_predictions']
        
        # Get the unique labels present in the evaluation, map them to names
        novel_labels = {config.class_to_label[c] for c in config.NOVEL_CLASSES}
        present_labels = sorted(list(set(all_labels + all_preds)))
        cm_labels = [l for l in present_labels if l in novel_labels or l == -1]
        class_names = [config.label_to_class.get(l, 'Unknown') for l in cm_labels]

        cm = confusion_matrix(all_labels, all_preds, labels=cm_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        plot_paths['Confusion Matrix'] = cm_path
    except Exception as e:
        print(f"⚠️ Could not generate confusion matrix plot: {e}")

    # ROC Curve
    try:
        fpr, tpr = roc_data
        if fpr is not None and tpr is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {metrics["auroc"]:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Open-Set Recognition (Known vs. Unknown)')
            plt.legend(loc="lower right")
            roc_path = "roc_curve.png"
            plt.savefig(roc_path, bbox_inches='tight')
            plt.close()
            plot_paths['ROC Curve'] = roc_path
    except Exception as e:
        print(f"⚠️ Could not generate ROC curve plot: {e}")

    # --- Create PDF ---
    pdf = PDF()
    
    # Title Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.ln(60)
    pdf.cell(0, 10, 'Remote Sensing Analysis', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, 'Performance Report', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, time.strftime("%Y-%m-%d %H:%M:%S"), 0, 1, 'C')
    
    # Table of Contents
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Table of Contents', 0, 1, 'L')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 12)
    sections = [
        "1. Executive Summary",
        "2. System Configuration",
        "3. Performance Metrics",
        "4. Visualization Results",
        "5. Per-Class Analysis",
        "6. Qualitative Examples"
    ]
    for section in sections:
        pdf.cell(0, 8, section, 0, 1, 'L')
    
    # 1. Executive Summary
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '1. Executive Summary', 0, 1, 'L')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 11)
    summary_text = f"""
This report presents the results of an automated remote sensing analysis using few-shot learning and open-set recognition techniques. The system achieved:

- Few-shot Classification Accuracy: {metrics.get('few_shot_accuracy_mean', 0):.3f} ± {metrics.get('few_shot_accuracy_std', 0):.3f}
- Open-Set Recognition AUROC: {metrics.get('auroc', 0.5):.3f}
- Unified Detection F1 Score: {metrics.get('udf1_score', 0):.3f}
- Total Processing Time: {time.time() - start_time:.2f} seconds

The analysis was performed on {len(config.NOVEL_CLASSES)} novel classes with {config.N_WAY}-way, {config.K_SHOT}-shot learning configuration.
    """
    pdf.multi_cell(0, 8, summary_text.strip(), align='L')
    
    # 2. System Configuration
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '2. System Configuration', 0, 1, 'L')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 11)
    
    config_data = {
        "Operation Mode": "DEMO" if config.DEMO_MODE else "FULL",
        "Base Model": config.MODEL_NAME,
        "Computing Device": config.DEVICE,
        "Task Configuration": f"{config.N_WAY}-way, {config.K_SHOT}-shot",
        "Model Enhancements": "- LoRA: " + ('Enabled' if config.LORA_ENABLED else 'Disabled') + "\n- CoOp: " + ('Enabled' if config.COOP_ENABLED else 'Disabled'),
        "Image Processing": "- Tile Sizes: " + str(config.TILE_SIZES) + "\n- Overlap Ratios: " + str(config.TILE_OVERLAP),
        "OSR Configuration": "- Calibration Method: " + config.OSR_CALIBRATION_METHOD + "\n- Percentile: " + str(config.OSR_PERCENTILE)
    }
    
    for key, value in config_data.items():
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, key, 0, 1, 'L')
        pdf.set_font('Helvetica', '', 11)
        pdf.multi_cell(0, 6, str(value), align='L')
        pdf.ln(2)
    
    # 3. Performance Metrics
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '3. Performance Metrics', 0, 1, 'L')
    pdf.ln(5)
    
    # Create a table for metrics
    pdf.set_font('Helvetica', 'B', 10)
    col_width = pdf.w / 4
    pdf.cell(col_width, 10, 'Metric', 1, 0, 'C')
    pdf.cell(col_width, 10, 'Value', 1, 0, 'C')
    pdf.cell(col_width * 2, 10, 'Description', 1, 1, 'C')
    
    pdf.set_font('Helvetica', '', 10)
    metrics_data = [
        ('Few-shot Accuracy', f"{metrics.get('few_shot_accuracy_mean', 0):.3f}", 'Classification accuracy on known novel classes'),
        ('AUROC', f"{metrics.get('auroc', 0.5):.3f}", 'Area under ROC curve for open-set detection'),
        ('UDF1 Score', f"{metrics.get('udf1_score', 0):.3f}", 'Unified metric combining known and unknown detection performance')
    ]
    
    for metric, value, desc in metrics_data:
        pdf.cell(col_width, 10, metric, 1, 0, 'L')
        pdf.cell(col_width, 10, value, 1, 0, 'C')
        pdf.multi_cell(col_width * 2, 10, desc, align='L')
    
    # 4. Visualization Results
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '4. Visualization Results', 0, 1, 'L')
    pdf.ln(5)
    
    # Add plots with proper spacing
    if 'Confusion Matrix' in plot_paths:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, 'Confusion Matrix', 0, 1, 'L')
        pdf.image(plot_paths['Confusion Matrix'], x=20, w=170)
        pdf.ln(10)
    
    if 'ROC Curve' in plot_paths:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, 'ROC Curve', 0, 1, 'L')
        pdf.image(plot_paths['ROC Curve'], x=20, w=170)
    
    # 5. Per-Class Analysis
    if 'per_class_data' in metrics and metrics['per_class_data']:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, '5. Per-Class Analysis', 0, 1, 'L')
        pdf.ln(5)
        
        # Table header with adjusted column widths
        pdf.set_font('Helvetica', 'B', 8)
        col_widths = {
            'class': 30,
            'metrics': 20,
            'counts': 15
        }
        
        headers = ['Class', 'Total', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
        widths = [col_widths['class']] + [col_widths['counts']] * 4 + [col_widths['metrics']] * 3
        
        for header, width in zip(headers, widths):
            pdf.cell(width, 7, header, 1, 0, 'C')
        pdf.ln()
        
        # Table content
        pdf.set_font('Helvetica', '', 8)
        for class_name in sorted(metrics['per_class_data'].keys()):
            data = metrics['per_class_data'][class_name]
            tp = data['true_positives']
            fp = data['false_positives']
            fn = data['false_negatives']
            total = data['total']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            row_data = [
                class_name, str(total), str(tp), str(fp), str(fn),
                f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}"
            ]
            
            for value, width in zip(row_data, widths):
                pdf.cell(width, 7, value, 1, 0, 'C')
            pdf.ln()
    
    # 6. Qualitative Examples
    if qualitative_results:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, '6. Qualitative Examples', 0, 1, 'L')
        pdf.ln(5)
        
        # Group examples by class
        class_examples = defaultdict(lambda: {'correct': [], 'incorrect': []})
        for res in qualitative_results:
            true_class = config.label_to_class.get(res['true_label'], 'Unknown')
            pred_class = config.label_to_class.get(res['pred_label'], 'Unknown')
            is_correct = res['true_label'] == res['pred_label']
            
            if is_correct and len(class_examples[true_class]['correct']) < 1:
                class_examples[true_class]['correct'].append(res)
            elif not is_correct and len(class_examples[true_class]['incorrect']) < 1:
                class_examples[true_class]['incorrect'].append(res)
        
        # Display examples in a grid format
        img_size = 80
        margin = 10
        examples_per_page = 4  # 2 classes per page (each with correct and incorrect)
        
        for class_name in sorted(class_examples.keys()):
            if len(pdf.pages) > 0 and pdf.get_y() > pdf.h - 150:  # Check if we need a new page
                pdf.add_page()
            
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 10, f'Class: {class_name}', 0, 1, 'L')
            pdf.ln(5)
            
            # Get examples for this class
            correct_examples = class_examples[class_name]['correct']
            incorrect_examples = class_examples[class_name]['incorrect']
            
            # Display correct example
            if correct_examples:
                res = correct_examples[0]
                x = pdf.l_margin
                y = pdf.get_y()
                
                # Save and add image
                img_buf = io.BytesIO()
                res['image'].resize((int(img_size), int(img_size))).save(img_buf, format='PNG')
                img_buf.seek(0)
                
                pdf.image(img_buf, x=x, y=y, w=img_size, h=img_size)
                
                # Add text below image
                pdf.set_xy(x, y + img_size + 2)
                pdf.set_font('Helvetica', '', 8)
                pdf.multi_cell(img_size, 4,
                             f"True: {class_name}\n"
                             f"Pred: {class_name}\n"
                             f"Conf: {res['confidence']:.2f}\n"
                             f"Status: Correct",
                             align='L')
            
            # Display incorrect example
            if incorrect_examples:
                res = incorrect_examples[0]
                x = pdf.l_margin + img_size + margin + 20
                y = pdf.get_y() - img_size - 10  # Align with correct example
                
                # Save and add image
                img_buf = io.BytesIO()
                res['image'].resize((int(img_size), int(img_size))).save(img_buf, format='PNG')
                img_buf.seek(0)
                
                pdf.image(img_buf, x=x, y=y, w=img_size, h=img_size)
                
                # Add text below image
                pdf.set_xy(x, y + img_size + 2)
                pdf.set_font('Helvetica', '', 8)
                pred_class = config.label_to_class.get(res['pred_label'], 'Unknown')
                pdf.multi_cell(img_size, 4,
                             f"True: {class_name}\n"
                             f"Pred: {pred_class}\n"
                             f"Conf: {res['confidence']:.2f}\n"
                             f"Status: Incorrect",
                             align='L')
            
            pdf.ln(img_size + 20)  # Add space after each class's examples

    try:
        pdf.output(report_filename)
        print(f"✅ Enhanced report saved as {report_filename}")
        # Clean up plots
        for path in plot_paths.values():
            os.remove(path)
    except Exception as e:
        print(f"⚠️ Report generation failed: {e}")
        return None
    return report_filename

# === SECTION: MAIN EXECUTION ===
def main():
    """Main execution pipeline for the remote sensing analyst."""
    start_time = time.time()
    try:
        print("="*60)
        print("🚀 CONSOLIDATED INTELLIGENT REMOTE SENSING ANALYST")
        print("="*60)

        device = setup_environment(CONFIG)
        dataset = EnhancedDiorDataset(CONFIG)
        model, processor, coop_module = load_enhanced_clip_model(CONFIG, device)
        
        # Run adaptation step if LoRA is enabled
        if CONFIG.LORA_ENABLED:
            run_adaptation(model, processor, dataset, CONFIG)

        inference_engine = EnhancedInference(model, processor, coop_module, CONFIG, device)
        osr_threshold = calibrate_osr_threshold(inference_engine, dataset, device, CONFIG)
        metrics, roc_data, qualitative_results = evaluate_performance(inference_engine, dataset, osr_threshold, CONFIG)
        report_filename = generate_enhanced_report(CONFIG, metrics, roc_data, qualitative_results, start_time)

        # Final Summary
        print("\n" + "="*60)
        print("🎉 EXECUTION COMPLETED!")
        print("="*60)
        print(f"⏱️  Total Runtime: {time.time() - start_time:.2f}s")
        print(f"📊 Accuracy (FS): {metrics.get('few_shot_accuracy_mean', 0):.3f}")
        print(f"📈 AUROC (OSR):   {metrics.get('auroc', 0.5):.3f}")
        print(f"🎯 UDF1 Score:    {metrics.get('udf1_score', 0):.3f}")
        if report_filename:
            print(f"📄 Report saved to: {os.path.abspath(report_filename)}")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n❌ EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)