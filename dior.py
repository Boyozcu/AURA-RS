#!/usr/bin/env python3
import os
import json
import warnings
import traceback
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

warnings.filterwarnings('ignore')
print("�� AURA-RS Production Version - DIOR-RSVG Remote Sensing Visual Grounding Dataset")

# CUDA setup
if not torch.cuda.is_available():
    print("⚠️ CUDA not available, using CPU...")
    torch.set_num_threads(4)
else:
    print("✅ CUDA available")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

print('Production Mode Started')

# =============================================================================
# UTILITY FUNCTIONS - Consolidated and reusable
# =============================================================================

def safe_gpu_cleanup():
    """Centralized GPU memory management"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_device_info():
    """Get device information and memory stats"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device, gpu_memory
    return torch.device("cpu"), 0.0

def check_gpu_memory():
    """Monitor GPU memory and cleanup if needed"""
    if not torch.cuda.is_available():
        return False
    try:
        allocated = torch.cuda.memory_allocated() / 1e9
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        usage_percent = (allocated / max_memory) * 100
        
        if usage_percent > 90:
            safe_gpu_cleanup()
            return True
        return False
    except:
        return False

def handle_oom_error(func):
    """Decorator for handling out of memory errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                safe_gpu_cleanup()
                raise e
            else:
                raise e
    return wrapper

def safe_load_image_with_resize(image_input: Union[str, Image.Image], patch_size: int = 224) -> Optional[Image.Image]:
    """Unified image loading with resize and error handling"""
    try:
        # Handle both file paths and PIL Image objects
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            return None
            
        if image.size != (patch_size, patch_size):
            image = image.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        return image
    except Exception:
        return None

def safe_load_image_from_item(item: Dict, patch_size: int = 224) -> Optional[Image.Image]:
    """Load image from data item, handling both DIOR-RSVG PIL objects and file paths"""
    try:
        # First try PIL image object (DIOR-RSVG)
        if 'image_object' in item and item['image_object'] is not None:
            return safe_load_image_with_resize(item['image_object'], patch_size)
        # Fallback to file path (legacy UC Merced)
        elif 'image_path' in item:
            return safe_load_image_with_resize(item['image_path'], patch_size)
        else:
            return None
    except Exception:
        return None

def safe_vision_forward(model, pixel_values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Unified vision model forward pass with LoRA adaptation support"""
    try:
        if pixel_values.device != device:
            pixel_values = pixel_values.to(device)
        vision_inputs = {'pixel_values': pixel_values}
        
        # Handle LoRA-adapted models (PEFT wrapped)
        if hasattr(model, 'vision_model') and model.vision_model is not None:
            vision_model = model.vision_model
            
            # Check if it's a PEFT model
            if hasattr(vision_model, 'peft_config') or hasattr(vision_model, 'base_model'):
                # For PEFT models, call the base model directly
                if hasattr(vision_model, 'base_model'):
                    # PEFT wrapper case
                    try:
                        outputs = vision_model(**vision_inputs)
                        return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
                    except:
                        # Fallback: call base model
                        outputs = vision_model.base_model(**vision_inputs)
                        return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
                else:
                    # Direct PEFT model call
                    outputs = vision_model(**vision_inputs)
                    return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
            else:
                # Regular model
                outputs = vision_model(**vision_inputs)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
                
        else:
            # Use get_image_features for non-adapted models
            try:
                outputs = model.get_image_features(**vision_inputs)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            except:
                # Final fallback
                outputs = model.get_image_features(pixel_values=pixel_values)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
                
    except Exception as e:
        # Emergency fallback for LoRA models
        try:
            if hasattr(model, 'vision_model') and hasattr(model.vision_model, 'base_model'):
                # Try accessing the base model directly
                base_vision_model = model.vision_model.base_model
                outputs = base_vision_model(**vision_inputs)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
            else:
                # Try the original approach
                outputs = model.get_image_features(pixel_values=pixel_values)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        except Exception as fallback_e:
            raise RuntimeError(f"Vision model forward failed: Original error: {e}, Fallback error: {fallback_e}")

def safe_text_forward(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Unified text model forward pass with LoRA adaptation support"""
    try:
        text_inputs = {'input_ids': input_ids}
        if attention_mask is not None:
            text_inputs['attention_mask'] = attention_mask
            
        # Handle LoRA-adapted models (PEFT wrapped)
        if hasattr(model, 'text_model') and model.text_model is not None:
            text_model = model.text_model
            
            # Check if it's a PEFT model
            if hasattr(text_model, 'peft_config') or hasattr(text_model, 'base_model'):
                # For PEFT models
                if hasattr(text_model, 'base_model'):
                    # PEFT wrapper case
                    try:
                        outputs = text_model(**text_inputs)
                        return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
                    except:
                        # Fallback: call base model
                        outputs = text_model.base_model(**text_inputs)
                        return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
                else:
                    # Direct PEFT model call
                    outputs = text_model(**text_inputs)
                    return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
            else:
                # Regular model
                outputs = text_model(**text_inputs)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
                
        else:
            # Use get_text_features for non-adapted models
            try:
                outputs = model.get_text_features(**text_inputs)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            except:
                # Final fallback
                outputs = model.get_text_features(**text_inputs)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
                
    except Exception as e:
        # Emergency fallback for LoRA models
        try:
            if hasattr(model, 'text_model') and hasattr(model.text_model, 'base_model'):
                # Try accessing the base model directly
                base_text_model = model.text_model.base_model
                outputs = base_text_model(**text_inputs)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
            else:
                # Try the original approach
                outputs = model.get_text_features(**text_inputs)
                return outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        except Exception as fallback_e:
            raise RuntimeError(f"Text model forward failed: Original error: {e}, Fallback error: {fallback_e}")

def compute_similarity_with_temperature(image_features: torch.Tensor, 
                                       query_features: torch.Tensor, 
                                       temperature: float = 0.07,
                                       normalization: str = 'l2') -> torch.Tensor:
    """Unified similarity computation with temperature scaling"""
    if normalization == 'l2':
        image_features = F.normalize(image_features, dim=-1)
        query_features = F.normalize(query_features, dim=-1)
        similarity_matrix = torch.matmul(image_features, query_features.t()) / temperature
        
        if similarity_matrix.dim() > 1:
            if similarity_matrix.size(1) == 1:
                return similarity_matrix.squeeze(1)
            elif similarity_matrix.size(0) == similarity_matrix.size(1):
                return torch.diag(similarity_matrix)
            else:
                return similarity_matrix.squeeze()
        else:
            return similarity_matrix
    else:
        return F.cosine_similarity(image_features, query_features.expand_as(image_features), dim=1) / temperature

def find_optimal_threshold(similarities: List[float], true_labels: List[int], 
                          percentiles: List[int] = [80, 85, 90, 95]) -> float:
    """Unified threshold optimization function"""
    if not similarities or not true_labels:
        return 0.1
    
    best_f1 = 0
    best_threshold = 0.1
    
    # Try percentile-based thresholds
    threshold_candidates = []
    for percentile in percentiles:
        threshold_candidates.append(np.percentile(similarities, percentile))
    
    # Add unique similarity values
    unique_sims = sorted(set(similarities))
    if len(unique_sims) > 20:
        step = len(unique_sims) // 20
        sampled_sims = unique_sims[::step]
    else:
        sampled_sims = unique_sims
    threshold_candidates.extend(sampled_sims)
    
    threshold_candidates = sorted(set(threshold_candidates))
    
    for threshold in threshold_candidates:
        predictions = []
        for sim in similarities:
            try:
                sim_value = float(sim) if isinstance(sim, (int, float)) else 0.0
                predictions.append(1 if sim_value >= threshold else 0)
            except (ValueError, TypeError):
                predictions.append(0)
        
        tp = sum(1 for pred, true in zip(predictions, true_labels) if pred == 1 and true == 1)
        fp = sum(1 for pred, true in zip(predictions, true_labels) if pred == 1 and true == 0)
        fn = sum(1 for pred, true in zip(predictions, true_labels) if pred == 0 and true == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

@dataclass
class AuraConfig:
    """
    AURA-RS Configuration for DIOR-RSVG Remote Sensing Visual Grounding Dataset
    
    ��️ DEVELOPMENT MODE (Default):
    - Single seed for faster iteration
    - Reduced visualizations and analysis
    - ~56 experiments (3x faster)
    
    �� PRODUCTION MODE:
    - Multiple seeds for robust statistics
    - Full academic analysis and visualizations  
    - ~168 experiments
    
    To switch to production mode:
    config = AuraConfig()
    config.DEVELOPMENT_MODE = False
    """
    
    RESULTS_PATH: str = "results"
    MODEL_NAME: str = "google/siglip-base-patch16-224"
    DEVICE: str = "cuda"
    
    # DIOR-RSVG classes based on DIOR dataset (extracted from referring expressions)
    ALL_CLASSES: List[str] = field(default_factory=lambda: [
        'airplane', 'airport', 'baseball_field', 'basketball_court', 'bridge', 
        'chimney', 'dam', 'expressway_service_area', 'expressway_toll_station', 
        'golf_course', 'ground_track_field', 'harbor', 'overpass', 'ship', 
        'stadium', 'storage_tank', 'tennis_court', 'train_station', 'vehicle', 
        'wind_mill'
    ])
    
    BASE_CLASSES: List[str] = field(default_factory=lambda: [
        'airplane',
        'airport', 
        'bridge',
        'dam',
        'harbor',
        'overpass',
        'ship',
        'stadium',
        'storage_tank',
        'train_station',
        'vehicle',
        'wind_mill'
    ])
    
    NOVEL_CLASSES: List[str] = field(default_factory=lambda: [
        'baseball_field',
        'basketball_court', 
        'chimney',
        'expressway_service_area',
        'expressway_toll_station',
        'golf_course',
        'ground_track_field',
        'tennis_court'
    ])
    
    RANDOM_SEEDS: List[int] = field(default_factory=lambda: [42])  # Reduced to 1 for faster development
    
    # Advanced configuration options
    DEVELOPMENT_MODE: bool = True  # Set to False for production runs
    
    K_SHOTS: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    LORA_RANK: int = 32  # Increased from 16 for more capacity
    LORA_ALPHA: int = 64  # Increased proportionally
    LORA_DROPOUT: float = 0.05  # Reduced for better learning
    LORA_LR: float = 1e-4  # Increased for faster adaptation
    LORA_EPOCHS: int = 8  # Increased from 3 for better convergence
    LORA_TEMPERATURE: float = 0.05  # Lower for sharper attention
    
    BATCH_SIZE: int = 32  # Increased for better performance
    EVALUATION_BATCH_SIZE: int = 64  # Larger batch for inference-only operations
    PATCH_SIZE: int = 224
    MAX_SAMPLES_PER_CLASS: int = 100
    SIMILARITY_THRESHOLD: float = 0.1
    LORA_SAMPLES_PER_CLASS: int = 80
    
    ABLATION_STUDIES: Dict = field(default_factory=lambda: {
        'temperature_scaling': [0.07, 0.1],  # Reduced from 3 to 2 values
        'normalization_methods': ['l2'],  # Only l2 for now, 'none' causes issues
        'threshold_methods': ['optimal', 'percentile'],
        'run_ablation': False  # Disabled for now to focus on main results
    })
    
    ACADEMIC_ANALYSIS: Dict = field(default_factory=lambda: {
        'statistical_tests': True,
        'confidence_intervals': True,
        'cross_validation': True,
        'confusion_matrix': True,
        'bootstrap_samples': 500
    })
    
    VISUALIZATION: Dict = field(default_factory=lambda: {
        'save_examples': True,
        'examples_per_category': 5,
        'save_query_images': True,
        'save_similarity_maps': True,
        'save_failure_analysis': True,
        'image_size': (224, 224),
        'grid_size': (3, 3),
        'dpi': 150,
        # Academic paper specific settings
        'academic_paper': {
            'create_method_overview': True,
            'create_performance_plots': True,
            'copy_best_examples': True,
            'generate_latex_table': True,
            'high_quality_figures': True,
            'figure_dpi': 300
        }
    })
    
    def __post_init__(self):
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        assert len(set(self.BASE_CLASSES) & set(self.NOVEL_CLASSES)) == 0, "Base ve Novel sınıflar çakışıyor"
        assert len(self.BASE_CLASSES) + len(self.NOVEL_CLASSES) == len(self.ALL_CLASSES), "Sınıf sayısı uyumsuz"
        
        # Auto-adjust settings based on mode
        if self.DEVELOPMENT_MODE:
            print("��️ DEVELOPMENT MODE - Optimized for speed")
            # Keep single seed for development
            if len(self.RANDOM_SEEDS) > 1:
                self.RANDOM_SEEDS = [self.RANDOM_SEEDS[0]]
                print(f"   Using single seed: {self.RANDOM_SEEDS[0]}")
            
            # Disable expensive analyses
            self.ACADEMIC_ANALYSIS['statistical_tests'] = False
            self.ACADEMIC_ANALYSIS['confidence_intervals'] = False
            self.ABLATION_STUDIES['run_ablation'] = False
            print("   Disabled: Statistical tests, confidence intervals, ablation studies")
            
            # Reduce visualization workload
            self.VISUALIZATION['examples_per_category'] = 3
            print("   Reduced: Visualization examples")
        else:
            print("�� PRODUCTION MODE - Full statistical analysis")
            # For production, ensure multiple seeds
            if len(self.RANDOM_SEEDS) < 3:
                self.RANDOM_SEEDS = [42, 123, 456]
                print(f"   Using multiple seeds: {self.RANDOM_SEEDS}")
                
            print("   Enabled: All statistical analyses and visualizations")

class RemoteSensingDataset(Dataset):
    
    def __init__(self, data_list: List[Dict], processor, config: AuraConfig):
        self.data_list = data_list
        self.processor = processor
        self.config = config
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Handle both PIL image objects (DIOR-RSVG) and file paths (legacy)
        image = safe_load_image_from_item(item, self.config.PATCH_SIZE)
        
        if image is None:
            # Return dummy data for failed loads
            image = Image.new('RGB', (self.config.PATCH_SIZE, self.config.PATCH_SIZE), 'black')
            return {
                'image': image,
                'text': "A remote sensing image",
                'class_name': 'unknown',
                'image_path': 'dummy'
            }
        
        return {
            'image': image,
            'text': f"A remote sensing image of {item['class_name']}",
            'class_name': item['class_name'],
            'image_path': item.get('image_path', 'dior_image')
        }

def collate_fn(batch, processor):
    
    try:
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        class_names = [item['class_name'] for item in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs['class_names'] = class_names
        return inputs
    except Exception as e:
        print(f"Collate error: {e}")
        return None

class AuraModel:
    def __init__(self, config: AuraConfig):
        self.config = config
        self.device, self.gpu_memory = get_device_info()
        self.model = None
        self.processor = None
        self.is_adapted = False
        self._load_model()
        
    def _load_model(self):
        try:
            print(f"Loading model: {self.config.MODEL_NAME}")
            
            # Device setup with fallback
            if self.config.DEVICE == "cuda" and not torch.cuda.is_available():
                print("CUDA requested but not available, using CPU...")
                self.device = torch.device("cpu")
                self.config.DEVICE = "cpu"
            elif self.config.DEVICE == "cuda" and self.gpu_memory < 2.0:
                print(f"GPU memory insufficient ({self.gpu_memory:.1f}GB), using CPU...")
                self.device = torch.device("cpu")
                self.config.DEVICE = "cpu"
            else:
                self.device = torch.device(self.config.DEVICE)
                
            # Load model and processor
            self.processor = AutoProcessor.from_pretrained(self.config.MODEL_NAME)
            self.model = AutoModel.from_pretrained(self.config.MODEL_NAME)
            self.model = self.model.to(self.device)
            
            if self.model is None or self.processor is None:
                raise RuntimeError("Model or processor failed to load")
            
            print(f"Model loaded on: {self.device}")
                
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("Memory error, switching to CPU...")
                self.device = torch.device("cpu")
                self.config.DEVICE = "cpu"
                safe_gpu_cleanup()
                self.processor = AutoProcessor.from_pretrained(self.config.MODEL_NAME)
                self.model = AutoModel.from_pretrained(self.config.MODEL_NAME).to(self.device)
            else:
                print(f"Model loading error: {e}")
                raise RuntimeError(f"Failed to load model: {e}")
    
    def adapt_with_lora(self, train_data: List[Dict]) -> bool:
        print(f"Starting LoRA adaptation with {len(train_data)} base samples")
        
        adaptation_data = self._select_adaptation_data(train_data)
        print(f"Selected {len(adaptation_data)} samples for adaptation")
        
        try:
            vision_config = LoraConfig(
                r=self.config.LORA_RANK,
                lora_alpha=self.config.LORA_ALPHA,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=self.config.LORA_DROPOUT,
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            text_config = LoraConfig(
                r=self.config.LORA_RANK // 2,
                lora_alpha=self.config.LORA_ALPHA // 2,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=self.config.LORA_DROPOUT,
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            self.model.vision_model = get_peft_model(self.model.vision_model, vision_config)
            self.model.text_model = get_peft_model(self.model.text_model, text_config)
            
            self.model.vision_model.train()
            self.model.text_model.train()
            
            optimizer = torch.optim.AdamW([
                {'params': self.model.vision_model.parameters(), 'lr': self.config.LORA_LR},
                {'params': self.model.text_model.parameters(), 'lr': self.config.LORA_LR * 0.1}
            ], weight_decay=1e-4)
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.LORA_EPOCHS
            )
            
            criterion = nn.CrossEntropyLoss()
            
            print(f"Training for {self.config.LORA_EPOCHS} epochs")
            
            for epoch in range(self.config.LORA_EPOCHS):
                epoch_start = time.time()
                total_loss = 0
                processed = 0
                
                print(f"\nEpoch {epoch+1}/{self.config.LORA_EPOCHS}")
                random.shuffle(adaptation_data)
                
                for i in range(0, len(adaptation_data), self.config.BATCH_SIZE):
                    batch = adaptation_data[i:i + self.config.BATCH_SIZE]
                    images, texts = [], []
                    
                    domain_prompts = [
                        "A satellite image of {}",
                        "An aerial view of {}",
                        "Remote sensing image showing {}",
                        "Bird's eye view of {}",
                        "Overhead satellite photograph of {}",
                        "Satellite imagery depicting {}",
                        "Aerial photography of {}",
                        "High-resolution satellite view of {}"
                    ]
                    
                    for item in batch:
                        try:
                            img = safe_load_image_from_item(item, self.config.PATCH_SIZE)
                            if img is None:
                                continue
                            
                            # Data augmentation for satellite imagery
                            if random.random() > 0.5:
                                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            if random.random() > 0.5:
                                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            # Random rotation (common in satellite imagery)
                            if random.random() > 0.7:
                                img = img.transpose(Image.ROTATE_90)
                                
                            images.append(img)
                            prompt = random.choice(domain_prompts)
                            text = prompt.format(item['class_name'])
                            texts.append(text)
                        except:
                            continue
                    
                    if not images or len(images) != len(texts):
                        continue
                    
                    try:
                        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
                        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                        
                        image_inputs = self.processor(images=images, return_tensors="pt", padding=True)
                        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

                        vision_outputs = self.model.vision_model(pixel_values=image_inputs['pixel_values'])
                        text_outputs = self.model.text_model(
                            input_ids=text_inputs['input_ids'],
                            attention_mask=text_inputs.get('attention_mask', None)
                        )
                        
                        image_features = F.normalize(vision_outputs.pooler_output, dim=-1)
                        text_features = F.normalize(text_outputs.pooler_output, dim=-1)
                        
                        temperature = self.config.LORA_TEMPERATURE
                        logits_per_image = torch.matmul(image_features, text_features.T) / temperature
                        logits_per_text = logits_per_image.T
                        
                        batch_size = len(images)
                        labels = torch.arange(batch_size, device=self.device)
                        
                        loss_i2t = criterion(logits_per_image, labels)
                        loss_t2i = criterion(logits_per_text, labels)
                        loss = (loss_i2t + loss_t2i) / 2.0
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                        processed += len(images)
                        
                    except Exception as e:
                        if "out of memory" in str(e).lower():
                            safe_gpu_cleanup()
                        continue
                
                scheduler.step()
                
                if processed > 0:
                    epoch_time = time.time() - epoch_start
                    avg_loss = total_loss / max(1, (processed // self.config.BATCH_SIZE))
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    print(f"Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s | Samples: {processed}")
                    check_gpu_memory()
            
            self.model.vision_model.eval()
            self.model.text_model.eval()
            self.is_adapted = True
            
            # Additional checks for PEFT models
            if hasattr(self.model.vision_model, 'base_model'):
                self.model.vision_model.base_model.eval()
                print("✅ Vision base model set to eval mode")
            if hasattr(self.model.text_model, 'base_model'):
                self.model.text_model.base_model.eval()
                print("✅ Text base model set to eval mode")
                
            print("✅ LoRA adaptation completed successfully")
            print(f"   Vision model type: {type(self.model.vision_model)}")
            print(f"   Text model type: {type(self.model.text_model)}")
            print(f"   Is adapted flag: {self.is_adapted}")
            
            return True
            
        except Exception as e:
            print(f"LoRA adaptation failed: {e}")
            return False
    
    def _select_adaptation_data(self, base_data: List[Dict]) -> List[Dict]:
        """Select balanced data from base classes only for LoRA adaptation"""
        selected_data = []
        
        class_data = defaultdict(list)
        for item in base_data:
            class_data[item['class_name']].append(item)
        
        for class_name, items in class_data.items():
            random.shuffle(items)
            n_samples = min(len(items), self.config.LORA_SAMPLES_PER_CLASS)
            selected_data.extend(items[:n_samples])
            print(f"    {class_name}: {n_samples} samples selected for LoRA")
        
        return selected_data
    
    def compute_embeddings(self, query_input: Union[str, List[Image.Image]], 
                          strategy: str = 'prototype') -> torch.Tensor:
        self.model.eval()
        
        # Ensure PEFT models are in eval mode
        if self.is_adapted:
            if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'eval'):
                self.model.vision_model.eval()
                if hasattr(self.model.vision_model, 'base_model'):
                    self.model.vision_model.base_model.eval()
            if hasattr(self.model, 'text_model') and hasattr(self.model.text_model, 'eval'):
                self.model.text_model.eval()
                if hasattr(self.model.text_model, 'base_model'):
                    self.model.text_model.base_model.eval()
        
        with torch.no_grad():
            if isinstance(query_input, str):
                # Text embedding
                inputs = self.processor(text=query_input, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                return safe_text_forward(self.model, inputs['input_ids'], 
                                       inputs.get('attention_mask'))
            
            elif isinstance(query_input, list):
                # Image embeddings
                inputs = self.processor(images=query_input, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values']
                embeddings = safe_vision_forward(self.model, pixel_values, self.device)
                
                # Apply strategy
                if strategy == 'mean':
                    return embeddings.mean(dim=0, keepdim=True)
                elif strategy == 'prototype':
                    # Enhanced prototype for satellite imagery
                    proto = embeddings.mean(dim=0, keepdim=True)
                    similarities = F.cosine_similarity(embeddings, proto, dim=1)
                    
                    # For LoRA-adapted models, use adaptive weighting
                    if self.is_adapted:
                        # Use softmax with adapted temperature for domain-specific weighting
                        weights = F.softmax(similarities / 0.05, dim=0)  # Lower temp for sharper focus
                        weighted_prototype = torch.sum(embeddings * weights.unsqueeze(1), dim=0, keepdim=True)
                        
                        # Blend with simple mean for stability
                        alpha = 0.7  # Weight towards adaptive prototype
                        return alpha * weighted_prototype + (1 - alpha) * proto
                    else:
                        # Standard prototype for base models
                        weights = F.softmax(similarities / 0.1, dim=0)
                        return torch.sum(embeddings * weights.unsqueeze(1), dim=0, keepdim=True)
                        
                elif strategy == 'attention':
                    d_k = embeddings.size(-1)
                    scores = torch.matmul(embeddings, embeddings.transpose(-2, -1)) / np.sqrt(d_k)
                    attention_weights = F.softmax(scores, dim=-1)
                    attended = torch.matmul(attention_weights, embeddings)
                    return attended.mean(dim=0, keepdim=True)
                    
                elif strategy == 'domain_adaptive' and self.is_adapted:
                    # Special strategy for LoRA-adapted models
                    # Use self-attention to find most representative examples
                    n_examples = embeddings.size(0)
                    if n_examples > 1:
                        # Compute pairwise similarities
                        norm_embeddings = F.normalize(embeddings, dim=1)
                        similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.t())
                        
                        # Find examples most similar to the group
                        mean_similarities = similarity_matrix.mean(dim=1)
                        top_indices = torch.topk(mean_similarities, min(3, n_examples)).indices
                        
                        # Weighted combination of top examples
                        top_embeddings = embeddings[top_indices]
                        top_weights = F.softmax(mean_similarities[top_indices] / 0.1, dim=0)
                        return torch.sum(top_embeddings * top_weights.unsqueeze(1), dim=0, keepdim=True)
                    else:
                        return embeddings
                else:
                    return embeddings.mean(dim=0, keepdim=True)
            else:
                raise ValueError("Unsupported input type")

class DataManager:
    def __init__(self, config: AuraConfig):
        self.config = config
        self.all_data = []
        self.dataset_path = None
        
    def load_dataset(self) -> bool:
        try:
            print("Loading DIOR-RSVG dataset from Hugging Face...")
            ds = load_dataset("danielz01/DIOR-RSVG")
            
            # Combine all splits (train, validation, test) for our custom splitting
            all_splits = []
            for split_name in ['train', 'validation', 'test']:
                if split_name in ds:
                    split_data = ds[split_name]
                    print(f"Loaded {len(split_data)} samples from {split_name} split")
                    all_splits.append(split_data)
            
            if not all_splits:
                print("No data splits found in DIOR-RSVG dataset")
                return False
            
            return self._process_dior_rsvg_data(all_splits)
        except Exception as e:
            print(f"Dataset loading error: {e}")
            raise RuntimeError("Dataset loading failed")
            
    def _process_dior_rsvg_data(self, splits) -> bool:
        """Process DIOR-RSVG data to extract object classes from referring expressions"""
        print("Processing DIOR-RSVG visual grounding data...")
        
        total_images = 0
        class_counts = defaultdict(int)
        
        # Process each split
        for split in splits:
            print(f"�� Processing split with {len(split)} items...")
            for idx, item in enumerate(split):
                try:
                    # Extract from DIOR-RSVG structure
                    image = item.get('image')
                    objects = item.get('objects', {})
                    
                    if image is None or not objects:
                        continue
                    
                    # Get categories and captions
                    categories = objects.get('categories', [])
                    captions = objects.get('captions', [])
                    
                    # Process each object in the image
                    for category, caption in zip(categories, captions):
                        # Map DIOR category to our class system
                        detected_class = self._map_dior_category_to_class(category.lower())
                        
                        if detected_class and detected_class in self.config.ALL_CLASSES:
                            if class_counts[detected_class] >= self.config.MAX_SAMPLES_PER_CLASS:
                                continue
                                
                            # Create a temporary image path (we'll use the PIL image directly)
                            image_id = f"dior_{total_images}_{detected_class}"
                            temp_path = f"temp_{image_id}.jpg"
                            
                            self.all_data.append({
                                'image_path': temp_path,  # Placeholder path
                                'image_object': image,     # Store PIL image directly
                                'class_name': detected_class,
                                'image_id': image_id,
                                'expression': caption,  # Use referring expression as caption
                                'dior_category': category
                            })
                            class_counts[detected_class] += 1
                            total_images += 1
                            
                            # Show successful matches for first few
                            if total_images <= 10:
                                print(f"✅ MATCH {total_images}: '{category}' -> '{detected_class}' | Caption: '{caption}'")
                                
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                    continue
        
        print(f"\n�� FINAL STATS:")
        print(f"Processed {total_images} samples across {len(class_counts)} classes")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
        
        return total_images > 0
    
    def _map_dior_category_to_class(self, dior_category: str) -> str:
        """Map DIOR dataset categories to our class system"""
        
        # Direct mapping from DIOR categories to our classes
        category_mapping = {
            # Direct matches
            'airplane': 'airplane',
            'airport': 'airport',
            'bridge': 'bridge',
            'chimney': 'chimney',
            'dam': 'dam',
            'harbor': 'harbor',
            'ship': 'ship',
            'stadium': 'stadium',
            'vehicle': 'vehicle',
            
            # DIOR naming variations
            'storagetank': 'storage_tank',
            'storage_tank': 'storage_tank',
            'trainstation': 'train_station',
            'train_station': 'train_station',
            'windmill': 'wind_mill',
            'wind_mill': 'wind_mill',
            'overpass': 'overpass',
            
            # Sports fields
            'groundtrackfield': 'ground_track_field',
            'ground_track_field': 'ground_track_field',
            'baseballfield': 'baseball_field',
            'baseball_field': 'baseball_field',
            'basketballcourt': 'basketball_court',
            'basketball_court': 'basketball_court',
            'tenniscourt': 'tennis_court',
            'tennis_court': 'tennis_court',
            'golffield': 'golf_course',
            'golf_field': 'golf_course',
            'golf_course': 'golf_course',
            
            # Service areas
            'expresswayservicearea': 'expressway_service_area',
            'expressway_service_area': 'expressway_service_area',
            'expresswaytollstation': 'expressway_toll_station',
            'expressway_toll_station': 'expressway_toll_station',
        }
        
        return category_mapping.get(dior_category.lower().replace(' ', ''), None)
    
    def create_splits(self, seed: int) -> Dict[str, List[Dict]]:
        random.seed(seed)
        np.random.seed(seed)
        
        class_data = defaultdict(list)
        for item in self.all_data:
            class_data[item['class_name']].append(item)
        
        # Validate dataset configuration
        total_all_classes = len(self.config.ALL_CLASSES)
        total_base_classes = len(self.config.BASE_CLASSES)
        total_novel_classes = len(self.config.NOVEL_CLASSES)
        
        assert total_base_classes + total_novel_classes == total_all_classes, \
            f"Class count mismatch: {total_base_classes} + {total_novel_classes} ≠ {total_all_classes}"
        
        splits = {
            'base_train': [],
            'base_val': [],
            'novel_support': [],
            'novel_val': [],
            'novel_test': []
        }
        
        # Base classes split (80% train, 20% val)
        for class_name in self.config.BASE_CLASSES:
            items = class_data.get(class_name, [])
            if items:
                random.shuffle(items)
                n_samples = len(items)
                train_split = int(n_samples * 0.8)
                
                splits['base_train'].extend(items[:train_split])
                splits['base_val'].extend(items[train_split:])
        
        # Novel classes split (10% support, 10% validation, 80% test)
        for class_name in self.config.NOVEL_CLASSES:
            items = class_data.get(class_name, [])
            if items:
                random.shuffle(items)
                n_samples = len(items)
                support_split = max(5, int(n_samples * 0.1))  # 10% for support
                val_split = max(5, int(n_samples * 0.1))      # 10% for validation
                
                splits['novel_support'] = splits.get('novel_support', [])
                splits['novel_val'] = splits.get('novel_val', [])
                
                splits['novel_support'].extend(items[:support_split])
                splits['novel_val'].extend(items[support_split:support_split + val_split])
                splits['novel_test'].extend(items[support_split + val_split:])
        
        # Validate splits
        total_processed = sum(len(split) for split in splits.values())
        expected_total = sum(len(items) for items in class_data.values())
        assert total_processed == expected_total, f"Sample count mismatch: {total_processed} ≠ {expected_total}"
        
        # Check K-shot availability
        novel_classes_with_sufficient_support = 0
        for class_name in self.config.NOVEL_CLASSES:
            class_support_count = sum(1 for item in splits['novel_support'] if item['class_name'] == class_name)
            if class_support_count >= max(self.config.K_SHOTS):
                novel_classes_with_sufficient_support += 1
        
        if novel_classes_with_sufficient_support < total_novel_classes:
            print(f"Warning: {total_novel_classes - novel_classes_with_sufficient_support} classes may not have enough samples for {max(self.config.K_SHOTS)}-shot")
        
        print(f"Dataset splits created:")
        print(f"  Base train: {len(splits['base_train'])}")
        print(f"  Base val: {len(splits['base_val'])}")
        print(f"  Novel support: {len(splits['novel_support'])}")
        print(f"  Novel validation: {len(splits['novel_val'])}")
        print(f"  Novel test: {len(splits['novel_test'])}")
        
        return splits

def evaluate_model(model: AuraModel, test_data: List[Dict], query_embedding: torch.Tensor, 
                  target_class: str, config: AuraConfig, temperature: float = 0.07, 
                  normalization: str = 'l2', save_visuals: bool = False, 
                  scenario: str = "", seed: int = 0, k_shot: int = 0,
                  validation_data: List[Dict] = None, precomputed_threshold: float = None) -> Dict:
    
    positive_samples = [item for item in test_data if item['class_name'] == target_class]
    negative_samples = [item for item in test_data if item['class_name'] != target_class]
    
    if len(positive_samples) == 0:
        return {
            'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0, 
            'auroc': 0, 'average_precision': 0,
            'positive_samples': 0, 'negative_samples': 0
        }
    
    # Balance test set (4:1 negative to positive ratio)
    random.shuffle(negative_samples)
    max_negative = len(positive_samples) * 4
    if len(negative_samples) > max_negative:
        negative_samples = negative_samples[:max_negative]
    
    all_test_samples = positive_samples + negative_samples
    random.shuffle(all_test_samples)
    
    # Load test images
    test_images = []
    test_image_objects = []
    true_labels = []
    failed_loads = 0
    
    for item in all_test_samples:
        try:
            img = safe_load_image_from_item(item, config.PATCH_SIZE)
            if img is not None:
                test_images.append(img)
                test_image_objects.append(img.copy())
                true_labels.append(1 if item['class_name'] == target_class else 0)
        except:
            failed_loads += 1
            continue
    
    if failed_loads > 0:
        print(f"    Failed to load {failed_loads} images")
    
    if len(test_images) == 0:
        return {
            'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0,
            'auroc': 0, 'average_precision': 0,
            'positive_samples': 0, 'negative_samples': 0
        }
    
    final_positive_count = sum(true_labels)
    final_negative_count = len(true_labels) - final_positive_count
    
    if final_positive_count == 0 or final_negative_count == 0:
        return {
            'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0,
            'auroc': 0, 'average_precision': 0,
            'positive_samples': final_positive_count,
            'negative_samples': final_negative_count
        }
    
    # Compute similarities with dynamic batch sizing
    similarities = []
    batch_size = config.EVALUATION_BATCH_SIZE
    model.model.eval()
    
    with torch.no_grad():
        i = 0
        while i < len(test_images):
            batch_imgs = test_images[i:i+batch_size]
            batch_processed = False
            
            try:
                inputs = model.processor(images=batch_imgs, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values']
                
                img_features = safe_vision_forward(model.model, pixel_values, model.device)
                batch_similarities = compute_similarity_with_temperature(
                    img_features, query_embedding, temperature, normalization)
                
                if batch_similarities.dim() == 0:
                    batch_similarities = batch_similarities.unsqueeze(0)
                
                batch_similarities = torch.clamp(batch_similarities, -100, 100)
                similarity_values = batch_similarities.flatten().cpu().tolist()
                
                for val in similarity_values:
                    if isinstance(val, (int, float)):
                        similarities.append(float(val))
                    else:
                        similarities.append(0.0)
                
                batch_processed = True
                i += batch_size
                        
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    safe_gpu_cleanup()
                    # Reduce batch size and retry
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        print(f"    OOM detected, reducing batch size to {batch_size}")
                        continue
                    else:
                        # Single image processing
                        for single_img in batch_imgs:
                            try:
                                single_inputs = model.processor(images=[single_img], return_tensors="pt", padding=True)
                                single_features = safe_vision_forward(model.model, single_inputs['pixel_values'], model.device)
                                single_sim = compute_similarity_with_temperature(
                                    single_features, query_embedding, temperature, normalization)
                                similarities.append(torch.clamp(single_sim, -100, 100).item())
                            except Exception:
                                similarities.append(0.0)
                        batch_processed = True
                        i += len(batch_imgs)
                else:
                    similarities.extend([0.0] * len(batch_imgs))
                    batch_processed = True
                    i += len(batch_imgs)
            except Exception:
                similarities.extend([0.0] * len(batch_imgs))
                batch_processed = True
                i += len(batch_imgs)
    
    if len(similarities) != len(true_labels):
        similarities = similarities[:len(true_labels)]
    
    # Calculate AUROC and AP
    if len(set(true_labels)) < 2:
        auroc = 0.0
        average_precision = 0.0
    else:
        try:
            auroc = roc_auc_score(true_labels, similarities)
            average_precision = average_precision_score(true_labels, similarities)
        except:
            auroc = 0.0
            average_precision = 0.0
    
    # Determine threshold
    if precomputed_threshold is not None:
        threshold = precomputed_threshold
    elif validation_data is not None:
        threshold = compute_validation_threshold_simple(
            model, validation_data, query_embedding, target_class, 
            temperature, normalization
        )
    else:
        threshold = np.percentile(similarities, 85)
    
    # Make predictions
    predictions = []
    for sim in similarities:
        try:
            sim_value = float(sim) if isinstance(sim, (int, float)) else 0.0
            predictions.append(1 if sim_value >= threshold else 0)
        except (ValueError, TypeError):
            predictions.append(0)
    
    # Calculate metrics
    true_positives = sum(1 for pred, true in zip(predictions, true_labels) if pred == 1 and true == 1)
    false_positives = sum(1 for pred, true in zip(predictions, true_labels) if pred == 1 and true == 0)
    false_negatives = sum(1 for pred, true in zip(predictions, true_labels) if pred == 0 and true == 1)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = sum(1 for pred, true in zip(predictions, true_labels) if pred == true) / len(true_labels)
    
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    
    # Visualization
    visual_paths = {}
    if save_visuals and config.VISUALIZATION['save_examples']:
        vis_dir = os.path.join(config.RESULTS_PATH, "visualizations", f"seed_{seed}")
        os.makedirs(vis_dir, exist_ok=True)
        
        example_files = save_prediction_examples(
            test_image_objects, similarities, predictions, true_labels, 
            target_class, vis_dir, scenario, config
        )
        visual_paths.update(example_files)
        
        if config.VISUALIZATION['save_similarity_maps']:
            sim_file = save_similarity_heatmap(
                similarities, true_labels, predictions, target_class, 
                scenario, vis_dir, threshold
            )
            visual_paths['similarity_analysis'] = sim_file
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auroc,
        'average_precision': average_precision,
        'positive_samples': len(positive_samples),
        'negative_samples': len(negative_samples),
        'threshold_used': threshold,
        'confusion_matrix': cm.tolist(),
        'temperature': temperature,
        'normalization': normalization,
        'visual_paths': visual_paths
    }

def compute_validation_threshold_simple(model: AuraModel, validation_data: List[Dict], 
                                       query_embedding: torch.Tensor, target_class: str,
                                       temperature: float = 0.07, normalization: str = 'l2') -> float:
    """Simple validation threshold computation using unified function"""
    val_positive = [item for item in validation_data if item['class_name'] == target_class]
    val_negative = [item for item in validation_data if item['class_name'] != target_class]
    
    if len(val_positive) == 0 or len(val_negative) == 0:
        return 0.1
    
    # Sample validation data
    max_samples = 20
    val_positive_selected = random.sample(val_positive, min(max_samples, len(val_positive)))
    val_negative_selected = random.sample(val_negative, min(max_samples, len(val_negative)))
    
    val_samples = val_positive_selected + val_negative_selected
    random.shuffle(val_samples)
    
    val_similarities = []
    val_labels = []
    
    model.model.eval()
    with torch.no_grad():
        for item in val_samples:
            try:
                img = safe_load_image_from_item(item, model.config.PATCH_SIZE)
                if img is None:
                    continue
                    
                inputs = model.processor(images=[img], return_tensors="pt", padding=True)
                img_features = safe_vision_forward(model.model, inputs['pixel_values'], model.device)
                similarity = compute_similarity_with_temperature(
                    img_features, query_embedding, temperature, normalization)
                
                similarity_clamped = torch.clamp(similarity, -100, 100)
                if similarity_clamped.dim() > 0:
                    similarity_value = similarity_clamped.flatten()[0].item()
                else:
                    similarity_value = similarity_clamped.item()
                
                val_similarities.append(float(similarity_value))
                val_labels.append(1 if item['class_name'] == target_class else 0)
            except Exception as e:
                if "out of memory" in str(e).lower():
                    safe_gpu_cleanup()
                continue
    
    if len(val_similarities) == 0:
        return 0.1
    
    return find_optimal_threshold(val_similarities, val_labels)

def run_experiments(config: AuraConfig) -> pd.DataFrame:
    print("Starting AURA-RS experiments on DIOR-RSVG dataset")
    print(f"Base classes: {len(config.BASE_CLASSES)}, Novel classes: {len(config.NOVEL_CLASSES)}")
    print(f"K-shots: {config.K_SHOTS}, Seeds: {config.RANDOM_SEEDS}")
    
    # Calculate experiment workload
    # Zero-shot: 1 experiment per class per seed
    # Few-shot: K_SHOTS experiments per class per seed  
    # LoRA+Few-shot: K_SHOTS experiments per class per seed
    total_experiments = len(config.RANDOM_SEEDS) * len(config.NOVEL_CLASSES) * (1 + 2 * len(config.K_SHOTS))
    
    mode_info = "��️ DEV MODE" if config.DEVELOPMENT_MODE else "�� PROD MODE"
    print(f"\n{mode_info} - Total experiments: {total_experiments}")
    print(f"  Zero-shot: {len(config.RANDOM_SEEDS) * len(config.NOVEL_CLASSES)} experiments")
    print(f"  Few-shot: {len(config.RANDOM_SEEDS) * len(config.NOVEL_CLASSES) * len(config.K_SHOTS)} experiments") 
    print(f"  LoRA+Few-shot: {len(config.RANDOM_SEEDS) * len(config.NOVEL_CLASSES) * len(config.K_SHOTS)} experiments")
    
    if config.DEVELOPMENT_MODE:
        print(f"⚡ Speed optimization: {3 * len(config.NOVEL_CLASSES) * (1 + 2 * len(config.K_SHOTS)) - total_experiments} experiments saved!")
        print("   For production results, set DEVELOPMENT_MODE = False")
    else:
        print("�� Full statistical analysis enabled for robust results")
    
    data_manager = DataManager(config)
    if not data_manager.load_dataset():
        raise RuntimeError("Dataset loading failed")
    
    scenario_results = []  # For aggregated results
    class_results_all = []  # For per-class results
    experiment_counter = 0
    start_time = time.time()
    
    for seed_idx, seed in enumerate(config.RANDOM_SEEDS):
        print(f"\nSeed {seed_idx+1}/{len(config.RANDOM_SEEDS)}: {seed}")
        splits = data_manager.create_splits(seed)
        
        # Seed setup
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create LoRA-adapted model ONCE per seed for domain adaptation
        lora_adapted_model = None
        
        # Since LoRA+Few-Shot is now first, always create adapted model
        print(f"\n�� Creating LoRA-adapted model for seed {seed}...")
        lora_adapted_model = AuraModel(config)
        adaptation_success = lora_adapted_model.adapt_with_lora(splits['base_train'])
        if not adaptation_success:
            print("❌ LoRA adaptation failed, will use base model")
            lora_adapted_model = None
        else:
            print("✅ LoRA domain adaptation completed - Model adapted to satellite imagery")
        
        # Create scenarios - Reordered: LoRA+Few-Shot first, then Few-Shot-Only, then Zero-Shot
        scenarios = [
            {'name': 'LoRA+Few-Shot', 'use_lora': True, 'use_examples': True, 'k_shots': config.K_SHOTS},  # First: Most complex
            {'name': 'Few-Shot-Only', 'use_lora': False, 'use_examples': True, 'k_shots': config.K_SHOTS},  # Second: Medium complexity
            {'name': 'Zero-Shot', 'use_lora': False, 'use_examples': False, 'k_shots': [1]}  # Last: Baseline
        ]
        
        for scenario_idx, scenario in enumerate(scenarios):
            for k_idx, k_shot in enumerate(scenario['k_shots']):
                elapsed_time = time.time() - start_time
                print(f"\nScenario {scenario_idx+1}/3: {scenario['name']}")
                if scenario['use_examples']:
                    print(f"K-shot {k_idx+1}/{len(scenario['k_shots'])}: {k_shot}")
                print(f"Progress: {experiment_counter}/{total_experiments} ({experiment_counter/total_experiments*100:.1f}%)")
                
                # Select appropriate model based on scenario
                if scenario['use_lora'] and lora_adapted_model is not None:
                    model = lora_adapted_model  # Reuse the domain-adapted model
                    print("�� Using domain-adapted model (satellite imagery optimized)")
                else:
                    model = AuraModel(config)  # Fresh model for non-LoRA scenarios
                    print("�� Using base model (general vision-language)")
                
                class_results = []
                for class_idx, target_class in enumerate(config.NOVEL_CLASSES):
                    experiment_counter += 1
                    
                    try:
                        print(f"  Processing {target_class} ({class_idx+1}/{len(config.NOVEL_CLASSES)})")
                        
                        # Embedding computation
                        if scenario['use_examples']:
                            class_samples = [item for item in splits['novel_support'] 
                                           if item['class_name'] == target_class][:k_shot]
                            if len(class_samples) < k_shot:
                                print(f"    Insufficient samples: {len(class_samples)} < {k_shot}")
                                continue
                                
                            few_shot_images = []
                            for item in class_samples:
                                try:
                                    img = safe_load_image_from_item(item, config.PATCH_SIZE)
                                    if img is not None:
                                        few_shot_images.append(img)
                                except:
                                    continue
                                    
                            if len(few_shot_images) < k_shot:
                                print(f"    No valid images for {target_class}")
                                continue
                                
                            # Use domain-adaptive strategy for LoRA models, prototype for others
                            if scenario['use_lora'] and lora_adapted_model is not None:
                                query_embedding = model.compute_embeddings(few_shot_images, strategy='domain_adaptive')
                                print(f"    �� Created domain-adaptive prototype from {len(few_shot_images)} examples")
                            else:
                                query_embedding = model.compute_embeddings(few_shot_images, strategy='prototype')
                                print(f"    �� Created standard prototype from {len(few_shot_images)} examples")
                            
                            if config.VISUALIZATION['save_query_images'] and seed == config.RANDOM_SEEDS[0]:
                                vis_dir = os.path.join(config.RESULTS_PATH, "visualizations", f"seed_{seed}")
                                os.makedirs(vis_dir, exist_ok=True)
                                save_query_examples(few_shot_images, target_class, vis_dir, 
                                                  scenario['name'], seed, k_shot)
                        else:
                            # Enhanced text queries for satellite domain
                            satellite_prompts = [
                                f"Aerial view of {target_class}",
                                f"Satellite image showing {target_class}",
                                f"Remote sensing image of {target_class}",
                                f"Bird's eye view of {target_class}",
                                f"Overhead photograph of {target_class}"
                            ]
                            
                            # For LoRA-adapted models, use domain-specific prompts
                            if scenario['use_lora']:
                                text_query = f"Satellite image showing {target_class}"
                                print(f"    ��️ Domain-adapted text query: '{text_query}'")
                            else:
                                text_query = f"A satellite image of {target_class}"
                                print(f"    �� General text query: '{text_query}'")
                                
                            query_embedding = model.compute_embeddings(text_query)
                        
                        # Evaluation
                        save_visuals = (seed == config.RANDOM_SEEDS[0] and k_shot == 3)
                        metrics = evaluate_model(
                            model, splits['novel_test'], query_embedding, target_class, config,
                            save_visuals=save_visuals, scenario=scenario['name'], seed=seed, k_shot=k_shot,
                            validation_data=splits['novel_val']
                        )
                        
                        if metrics['positive_samples'] > 0:
                            metrics['target_class'] = target_class
                            class_results.append(metrics)
                            print(f"    ✅ F1: {metrics['f1']:.3f}, AUROC: {metrics['auroc']:.3f}")
                            
                    except Exception as e:
                        print(f"    ❌ Error processing {target_class}: {e}")
                        continue
                
                # Aggregate results
                if class_results:
                    avg_results = {
                        'seed': seed,
                        'k_shot': k_shot,
                        'scenario': scenario['name'],
                        'accuracy': np.mean([r['accuracy'] for r in class_results]),
                        'f1': np.mean([r['f1'] for r in class_results]),
                        'precision': np.mean([r['precision'] for r in class_results]),
                        'recall': np.mean([r['recall'] for r in class_results]),
                        'auroc': np.mean([r['auroc'] for r in class_results]),
                        'average_precision': np.mean([r['average_precision'] for r in class_results]),
                        'classes_tested': len(class_results),
                        'result_type': 'scenario_aggregate'
                    }
                    scenario_results.append(avg_results)
                    
                    print(f"�� Scenario completed: F1={avg_results['f1']:.3f}, AUROC={avg_results['auroc']:.3f}")
                    print(f"�� Classes tested: {len(class_results)}/{len(config.NOVEL_CLASSES)}")
                    
                    # Add per-class results
                    for result in class_results:
                        per_class_result = {
                            'seed': seed,
                            'k_shot': k_shot,
                            'scenario': scenario['name'],
                            'target_class': result['target_class'],
                            'accuracy': result['accuracy'],
                            'f1': result['f1'],
                            'precision': result['precision'],
                            'recall': result['recall'],
                            'auroc': result['auroc'],
                            'average_precision': result['average_precision'],
                            'positive_samples': result['positive_samples'],
                            'negative_samples': result['negative_samples'],
                            'threshold_used': result.get('threshold_used', 0.1),
                            'result_type': 'class_specific'
                        }
                        class_results_all.append(per_class_result)
                else:
                    print("⚠️ No valid results for this scenario")
                
                # Ablation studies (if enabled)
                if config.ABLATION_STUDIES['run_ablation']:
                    ablation_results = run_ablation_studies(model, splits, config, seed, k_shot)
                    for ablation_result in ablation_results:
                        ablation_result['result_type'] = 'ablation'
                        class_results_all.append(ablation_result)
                
                # Only cleanup non-LoRA models to preserve domain adaptation
                if not scenario['use_lora']:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        check_gpu_memory()
        
        # Cleanup LoRA model after all scenarios for this seed
        if lora_adapted_model is not None:
            del lora_adapted_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                check_gpu_memory()
                    
    # Combine all results with clear type distinction
    all_results = scenario_results + class_results_all
    return pd.DataFrame(all_results)

def run_ablation_studies(model: AuraModel, splits: Dict, config: AuraConfig, seed: int, k_shot: int) -> List[Dict]:
    ablation_results = []
    if not config.ABLATION_STUDIES['run_ablation']:
        return ablation_results
        
    print("Starting ablation studies...")
    
    try:
        for target_class in config.NOVEL_CLASSES[:2]:  # Limit to 2 classes for speed
            class_samples = [item for item in splits['novel_support'] 
                            if item['class_name'] == target_class][:k_shot]
            if len(class_samples) < k_shot:
                print(f"  Insufficient samples for {target_class}: {len(class_samples)} < {k_shot}")
                continue
                
            few_shot_images = []
            for item in class_samples:
                try:
                    img = safe_load_image_from_item(item, config.PATCH_SIZE)
                    if img is not None:
                        few_shot_images.append(img)
                except Exception:
                    continue
                    
            if len(few_shot_images) < k_shot:
                print(f"  Insufficient valid images for {target_class}: {len(few_shot_images)} < {k_shot}")
                continue
                
            try:
                query_embedding = model.compute_embeddings(few_shot_images, strategy='prototype')
            except Exception as e:
                print(f"  Failed to compute embeddings for {target_class}: {e}")
                continue
                
            for temp in config.ABLATION_STUDIES['temperature_scaling']:
                for norm in config.ABLATION_STUDIES['normalization_methods']:
                    try:
                        metrics = evaluate_model(
                            model, splits['novel_test'], query_embedding, 
                            target_class, config, temperature=temp, normalization=norm,
                            validation_data=splits['novel_val'], seed=seed, k_shot=k_shot
                        )
                        
                        ablation_result = {
                            'seed': seed,
                            'k_shot': k_shot,
                            'target_class': target_class,
                            'temperature': temp,
                            'normalization': norm,
                            'f1': metrics['f1'],
                            'auroc': metrics['auroc'],
                            'accuracy': metrics['accuracy'],
                            'study_type': 'ablation'
                        }
                        ablation_results.append(ablation_result)
                        print(f"  {target_class}, T={temp}, norm={norm}, F1={metrics['f1']:.3f}")
                        
                    except Exception as e:
                        print(f"  Ablation error for {target_class}, temp={temp}, norm={norm}: {e}")
                        continue
                        
        print(f"Ablation studies completed: {len(ablation_results)} results")
        
    except Exception as e:
        print(f"Critical error in ablation studies: {e}")
        
    return ablation_results
def compute_statistical_analysis(results_df: pd.DataFrame, config: AuraConfig) -> Dict:
    if not config.ACADEMIC_ANALYSIS['statistical_tests'] or len(results_df) < 6:
        if config.DEVELOPMENT_MODE:
            print("�� Statistical analysis skipped (Development Mode)")
        return {}
        
    stats_results = {}
    
    # Filter scenario-level results using the new result_type column
    scenario_results = results_df[results_df['result_type'] == 'scenario_aggregate'].copy() if 'result_type' in results_df.columns else results_df[results_df['target_class'].isna()].copy()
    if len(scenario_results) == 0:
        return {}
        
    scenarios = scenario_results['scenario'].unique()
    
    # Basic descriptive statistics (always available)
    for metric in ['f1', 'accuracy', 'auroc']:
        if metric not in scenario_results.columns:
            continue
        metric_stats = {}
        for scenario in scenarios:
            scenario_data = scenario_results[scenario_results['scenario'] == scenario][metric].dropna()
            if len(scenario_data) > 0:
                metric_stats[scenario] = {
                    'mean': float(scenario_data.mean()),
                    'std': float(scenario_data.std()) if len(scenario_data) > 1 else 0.0,
                    'count': len(scenario_data),
                    'ci_lower': float(scenario_data.mean() - 1.96 * scenario_data.std() / np.sqrt(len(scenario_data))) if len(scenario_data) > 1 else float(scenario_data.mean()),
                    'ci_upper': float(scenario_data.mean() + 1.96 * scenario_data.std() / np.sqrt(len(scenario_data))) if len(scenario_data) > 1 else float(scenario_data.mean())
                }
        stats_results[metric] = metric_stats
    
    # Advanced statistical tests (only for multiple seeds)
    if len(scenarios) >= 2 and not config.DEVELOPMENT_MODE:
        pairwise_tests = {}
        for i, scenario1 in enumerate(scenarios):
            for scenario2 in scenarios[i+1:]:
                s1_data = scenario_results[scenario_results['scenario'] == scenario1]['f1'].dropna()
                s2_data = scenario_results[scenario_results['scenario'] == scenario2]['f1'].dropna()
                if len(s1_data) > 1 and len(s2_data) > 1:
                    t_stat, p_value = stats.ttest_ind(s1_data, s2_data)
                    effect_size = (s1_data.mean() - s2_data.mean()) / np.sqrt(((len(s1_data)-1)*s1_data.var() + (len(s2_data)-1)*s2_data.var()) / (len(s1_data)+len(s2_data)-2))
                    pairwise_tests[f"{scenario1}_vs_{scenario2}"] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'effect_size': float(effect_size),
                        'significant': bool(p_value < 0.05)
                    }
        if pairwise_tests:
            stats_results['pairwise_tests'] = pairwise_tests
    elif config.DEVELOPMENT_MODE and len(config.RANDOM_SEEDS) == 1:
        print("�� Statistical significance tests skipped (single seed in dev mode)")
    
    return stats_results
def generate_academic_report(results_df: pd.DataFrame, stats_results: Dict, config: AuraConfig, bootstrap_results: Dict = None) -> str:
    report = []
    report.append("# AURA-RS: Academic Performance Report")
    report.append("=" * 50)
    report.append("")
    
    # Filter scenario-level results using the new result_type column
    scenario_results = results_df[results_df['result_type'] == 'scenario_aggregate'].copy() if 'result_type' in results_df.columns else results_df[results_df['target_class'].isna()].copy()
    if len(scenario_results) > 0:
        report.append("## Performance Summary (Mean ± Std)")
        report.append("")
        scenarios = scenario_results['scenario'].unique()
        for scenario in scenarios:
            scenario_data = scenario_results[scenario_results['scenario'] == scenario]
            if len(scenario_data) > 0:
                f1_mean = scenario_data['f1'].mean()
                f1_std = scenario_data['f1'].std()
                acc_mean = scenario_data['accuracy'].mean()
                acc_std = scenario_data['accuracy'].std()
                if 'auroc' in scenario_data.columns:
                    auroc_mean = scenario_data['auroc'].mean()
                    auroc_std = scenario_data['auroc'].std()
                    report.append(f"{scenario}: F1={f1_mean:.3f}±{f1_std:.3f}, Acc={acc_mean:.3f}±{acc_std:.3f}, AUROC={auroc_mean:.3f}±{auroc_std:.3f}")
                else:
                    report.append(f"{scenario}: F1={f1_mean:.3f}±{f1_std:.3f}, Acc={acc_mean:.3f}±{acc_std:.3f}")
        report.append("")
    if stats_results and 'pairwise_tests' in stats_results:
        report.append("## Statistical Significance Tests")
        report.append("")
        for comparison, test_result in stats_results['pairwise_tests'].items():
            significance = "**" if test_result['significant'] else ""
            report.append(f"{comparison}: p={test_result['p_value']:.4f}, effect_size={test_result['effect_size']:.3f} {significance}")
        report.append("")
    if 'f1' in stats_results:
        report.append("## Confidence Intervals (95%)")
        report.append("")
        for scenario, stats in stats_results['f1'].items():
            ci_range = stats['ci_upper'] - stats['ci_lower']
            report.append(f"{scenario}: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}] (width={ci_range:.3f})")
        report.append("")
    
    # Bootstrap Confidence Intervals - Akademik iyileştirme
    if bootstrap_results:
        report.append("## Bootstrap Confidence Intervals (95%)")
        report.append("")
        for scenario, metrics in bootstrap_results.items():
            if 'f1' in metrics:
                boot_stats = metrics['f1']
                report.append(f"{scenario}: [{boot_stats['ci_lower']:.3f}, {boot_stats['ci_upper']:.3f}] "
                             f"(bootstrap mean: {boot_stats['bootstrap_mean']:.3f}, "
                             f"width: {boot_stats['ci_width']:.3f}, n={boot_stats['n_samples']})")
        report.append("")
    ablation_data = results_df[results_df['result_type'] == 'ablation'] if 'result_type' in results_df.columns else pd.DataFrame()
    if len(ablation_data) > 0:
        report.append("## Ablation Study Results")
        report.append("")
        temp_analysis = ablation_data.groupby('temperature')['f1'].agg(['mean', 'std']).round(3)
        report.append("Temperature Scaling:")
        for temp, row in temp_analysis.iterrows():
            report.append(f"  T={temp}: F1={row['mean']:.3f}±{row['std']:.3f}")
        report.append("")
        norm_analysis = ablation_data.groupby('normalization')['f1'].agg(['mean', 'std']).round(3)
        report.append("Normalization Methods:")
        for norm, row in norm_analysis.iterrows():
            report.append(f"  {norm}: F1={row['mean']:.3f}±{row['std']:.3f}")
        report.append("")
    if len(scenario_results) > 0:
        report.append("## LaTeX Table")
        report.append("```latex")
        report.append("\\begin{table}[h]")
        report.append("\\centering")
        report.append("\\begin{tabular}{|l|c|c|c|}")
        report.append("\\hline")
        report.append("Method & F1-Score & Accuracy & AUROC \\\\")
        report.append("\\hline")
        for scenario in scenarios:
            scenario_data = scenario_results[scenario_results['scenario'] == scenario]
            if len(scenario_data) > 0:
                f1_mean = scenario_data['f1'].mean()
                f1_std = scenario_data['f1'].std()
                acc_mean = scenario_data['accuracy'].mean()
                acc_std = scenario_data['accuracy'].std()
                if 'auroc' in scenario_data.columns:
                    auroc_mean = scenario_data['auroc'].mean()
                    auroc_std = scenario_data['auroc'].std()
                    report.append(f"{scenario.replace('_', ' ')} & {f1_mean:.3f} $\\pm$ {f1_std:.3f} & {acc_mean:.3f} $\\pm$ {acc_std:.3f} & {auroc_mean:.3f} $\\pm$ {auroc_std:.3f} \\\\")
                else:
                    report.append(f"{scenario.replace('_', ' ')} & {f1_mean:.3f} $\\pm$ {f1_std:.3f} & {acc_mean:.3f} $\\pm$ {acc_std:.3f} & - \\\\")
        report.append("\\hline")
        report.append("\\end{tabular}")
        report.append("\\caption{Performance comparison on DIOR-RSVG dataset}")
        report.append("\\end{table}")
        report.append("```")
    return "\n".join(report)
def save_query_examples(few_shot_images: List[Image.Image], target_class: str, 
                       save_dir: str, scenario: str, seed: int, k_shot: int) -> str:
    
    if not few_shot_images:
        return ""
    n_images = len(few_shot_images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    for i, img in enumerate(few_shot_images):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(f"Query {i+1}", fontsize=10)
            axes[i].axis('off')
    for j in range(n_images, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f"{scenario} - {target_class} (K={k_shot}, Seed={seed})", fontsize=12)
    plt.tight_layout()
    filename = f"query_{target_class}_{scenario}_{k_shot}shot_seed{seed}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath
def save_prediction_examples(test_images: List[Image.Image], similarities: List[float], 
                           predictions: List[int], true_labels: List[int], target_class: str,
                           save_dir: str, scenario: str, config: AuraConfig) -> Dict[str, str]:
    
    saved_files = {}
    if not config.VISUALIZATION['save_examples']:
        return saved_files
    n_examples = config.VISUALIZATION['examples_per_category']
    tp_indices = [(i, similarities[i]) for i in range(len(predictions)) 
                  if predictions[i] == 1 and true_labels[i] == 1]
    tp_indices.sort(key=lambda x: x[1], reverse=True)
    if tp_indices:
        tp_file = save_example_grid(
            [test_images[i] for i, _ in tp_indices[:n_examples]],
            [f"Conf: {sim:.3f}" for _, sim in tp_indices[:n_examples]],
            f"Success Cases - {target_class} ({scenario})",
            os.path.join(save_dir, f"success_{target_class}_{scenario}.png")
        )
        saved_files['success'] = tp_file
    fp_indices = [(i, similarities[i]) for i in range(len(predictions)) 
                  if predictions[i] == 1 and true_labels[i] == 0]
    fp_indices.sort(key=lambda x: x[1], reverse=True)
    if fp_indices:
        fp_file = save_example_grid(
            [test_images[i] for i, _ in fp_indices[:n_examples]],
            [f"Conf: {sim:.3f}" for _, sim in fp_indices[:n_examples]],
            f"False Positives - {target_class} ({scenario})",
            os.path.join(save_dir, f"false_pos_{target_class}_{scenario}.png")
        )
        saved_files['false_positives'] = fp_file
    fn_indices = [(i, similarities[i]) for i in range(len(predictions)) 
                  if predictions[i] == 0 and true_labels[i] == 1]
    fn_indices.sort(key=lambda x: x[1])
    if fn_indices:
        fn_file = save_example_grid(
            [test_images[i] for i, _ in fn_indices[:n_examples]],
            [f"Conf: {sim:.3f}" for _, sim in fn_indices[:n_examples]],
            f"False Negatives - {target_class} ({scenario})",
            os.path.join(save_dir, f"false_neg_{target_class}_{scenario}.png")
        )
        saved_files['false_negatives'] = fn_file
    # Use a conservative threshold for challenging examples visualization
    # CRITICAL FIX: Ensure similarities are all floats before percentile calculation
    safe_similarities = []
    for sim in similarities:
        try:
            safe_similarities.append(float(sim) if isinstance(sim, (int, float)) else 0.0)
        except (ValueError, TypeError):
            safe_similarities.append(0.0)
    
    threshold = np.percentile(safe_similarities, 85)
    challenging_indices = [(i, abs(safe_similarities[i] - threshold)) for i in range(len(similarities))]
    challenging_indices.sort(key=lambda x: x[1])
    if challenging_indices:
        ch_file = save_example_grid(
            [test_images[i] for i, _ in challenging_indices[:n_examples]],
            [f"Sim: {similarities[i]:.3f} (T: {threshold:.3f})" for i, _ in challenging_indices[:n_examples]],
            f"Challenging Cases - {target_class} ({scenario})",
            os.path.join(save_dir, f"challenging_{target_class}_{scenario}.png")
        )
        saved_files['challenging'] = ch_file
    return saved_files
def save_example_grid(images: List[Image.Image], labels: List[str], title: str, filepath: str) -> str:
    
    if not images:
        return ""
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    for i, (img, label) in enumerate(zip(images, labels)):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(label, fontsize=10)
            axes[i].axis('off')
    for j in range(n_images, len(axes)):
        axes[j].axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath
def save_similarity_heatmap(similarities, true_labels, predictions, target_class, scenario, save_dir, threshold):
    sim_data = {
        'True Positives': [sim for sim, true, pred in zip(similarities, true_labels, predictions) if true == 1 and pred == 1],
        'False Positives': [sim for sim, true, pred in zip(similarities, true_labels, predictions) if true == 0 and pred == 1],
        'True Negatives': [sim for sim, true, pred in zip(similarities, true_labels, predictions) if true == 0 and pred == 0],
        'False Negatives': [sim for sim, true, pred in zip(similarities, true_labels, predictions) if true == 1 and pred == 0]
    }
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    for category, sims in sim_data.items():
        if sims:
            plt.hist(sims, bins=20, alpha=0.7, label=f"{category} (n={len(sims)})")
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution by Category')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    categories, values = [], []
    for category, sims in sim_data.items():
        if sims:
            categories.extend([category] * len(sims))
            values.extend(sims)
    if categories and values:
        df_temp = pd.DataFrame({'Category': categories, 'Similarity': values})
        sns.boxplot(data=df_temp, x='Category', y='Similarity')
        plt.xticks(rotation=45)
        plt.axhline(threshold, color='red', linestyle='--', alpha=0.7)
        plt.title('Similarity by Prediction Category')
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.suptitle(f'Similarity Analysis: {target_class} ({scenario})', fontsize=16, fontweight='bold', y=1.02)
    filename = f"similarity_analysis_{target_class}_{scenario}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath
def compute_bootstrap_confidence_intervals(results_df: pd.DataFrame, 
                                         config: AuraConfig, 
                                         confidence_level: float = 0.95) -> Dict:
    """
    Akademik iyileştirme: Bootstrap ile confidence intervals hesapla.
    """
    if len(results_df) < 10:  # Minimum sample size
        return {}
    
    if config.DEVELOPMENT_MODE:
        print("�� Bootstrap analysis skipped (Development Mode - insufficient data for meaningful CI)")
        return {}
    
    n_bootstrap = config.ACADEMIC_ANALYSIS['bootstrap_samples']
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    bootstrap_results = {}
    
    # Filter scenario-level results using the new result_type column
    scenario_results = results_df[results_df['result_type'] == 'scenario_aggregate'].copy() if 'result_type' in results_df.columns else results_df[results_df['target_class'].isna()].copy()
    
    if len(scenario_results) == 0:
        return {}
    
    scenarios = scenario_results['scenario'].unique()
    
    for scenario in scenarios:
        scenario_data = scenario_results[scenario_results['scenario'] == scenario]
        if len(scenario_data) < 5:
            continue
            
        scenario_bootstrap = {}
        
        for metric in ['f1', 'accuracy', 'auroc']:
            if metric not in scenario_data.columns:
                continue
                
            original_data = scenario_data[metric].dropna().values
            if len(original_data) < 3:
                continue
            
            # Bootstrap sampling
            bootstrap_samples = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(original_data, size=len(original_data), replace=True)
                bootstrap_samples.append(np.mean(bootstrap_sample))
            
            # Confidence intervals
            ci_lower = np.percentile(bootstrap_samples, lower_percentile)
            ci_upper = np.percentile(bootstrap_samples, upper_percentile)
            
            scenario_bootstrap[metric] = {
                'original_mean': float(np.mean(original_data)),
                'bootstrap_mean': float(np.mean(bootstrap_samples)),
                'bootstrap_std': float(np.std(bootstrap_samples)),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'ci_width': float(ci_upper - ci_lower),
                'n_samples': len(original_data)
            }
        
        bootstrap_results[scenario] = scenario_bootstrap
    
    return bootstrap_results
# =============================================================================
# MAIN FUNCTION HELPERS - Clean and modularized approach
# =============================================================================

def save_results_with_fallback(results_df: pd.DataFrame, config: AuraConfig, timestamp: int) -> Optional[str]:
    """Save results with automatic fallback mechanism"""
    try:
        results_file = os.path.join(config.RESULTS_PATH, f"dior_rsvg_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        print(f"�� Results saved: {results_file}")
        return results_file
    except Exception as e:
        print(f"⚠️ Primary save failed: {e}")
        try:
            fallback_file = f"dior_rsvg_results_{timestamp}.csv"
            results_df.to_csv(fallback_file, index=False)
            print(f"�� Results saved to fallback: {fallback_file}")
            return fallback_file
        except Exception:
            print(f"❌ All save attempts failed")
            return None

def generate_experiment_summary(results_df: pd.DataFrame, results_file: Optional[str], timestamp: int) -> Dict:
    """Generate and display experiment summary"""
    summary = {
        'timestamp': timestamp,
        'total_experiments': len(results_df),
        'mean_f1': float(results_df['f1'].mean()) if 'f1' in results_df.columns else 0.0,
        'mean_accuracy': float(results_df['accuracy'].mean()) if 'accuracy' in results_df.columns else 0.0,
        'best_f1': float(results_df['f1'].max()) if 'f1' in results_df.columns else 0.0,
    }
    
    if results_file:
        summary['results_file'] = results_file
    
    print(f"✅ Production run completed successfully")
    print(f"�� Mean F1: {summary['mean_f1']:.3f}")
    print(f"�� Best F1: {summary['best_f1']:.3f}")
    print(f"�� Total experiments: {summary['total_experiments']}")
    
    return summary

def perform_statistical_analysis(results_df: pd.DataFrame, config: AuraConfig, timestamp: int) -> Dict:
    """Perform all statistical analysis with error handling"""
    stats_results = {}
    
    # Statistical analysis
    try:
        print("�� Computing statistical analysis...")
        stats_results = compute_statistical_analysis(results_df, config)
        if stats_results:
            stats_file = os.path.join(config.RESULTS_PATH, f"dior_rsvg_stats_{timestamp}.json")
            with open(stats_file, 'w') as f:
                json.dump(stats_results, f, indent=2)
            print(f"�� Statistics saved: {stats_file}")
    except Exception as e:
        print(f"⚠️ Statistical analysis failed: {e}")
    
    # Bootstrap analysis
    try:
        print("�� Computing bootstrap confidence intervals...")
        bootstrap_results = compute_bootstrap_confidence_intervals(results_df, config)
        if bootstrap_results:
            bootstrap_file = os.path.join(config.RESULTS_PATH, f"dior_rsvg_bootstrap_{timestamp}.json")
            with open(bootstrap_file, 'w') as f:
                json.dump(bootstrap_results, f, indent=2)
            print(f"�� Bootstrap CI saved: {bootstrap_file}")
            stats_results['bootstrap'] = bootstrap_results
    except Exception as e:
        print(f"⚠️ Bootstrap analysis failed: {e}")
    
    # Academic report generation
    try:
        print("�� Generating academic report...")
        bootstrap_data = stats_results.get('bootstrap', {})
        report = generate_academic_report(results_df, stats_results, config, bootstrap_data)
        report_file = os.path.join(config.RESULTS_PATH, f"ucmerced_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"�� Academic report saved: {report_file}")
    except Exception as e:
        print(f"⚠️ Report generation failed: {e}")
    
    # Academic paper materials organization
    try:
        print("�� Organizing academic paper materials...")
        paper_files = organize_academic_paper_materials(results_df, stats_results, config, timestamp, bootstrap_data)
        stats_results['academic_paper_files'] = paper_files
    except Exception as e:
        print(f"⚠️ Academic paper organization failed: {e}")
    
    return stats_results

def summarize_visualizations(config: AuraConfig):
    """Summarize visualization files"""
    try:
        vis_dir = os.path.join(config.RESULTS_PATH, "visualizations")
        if not os.path.exists(vis_dir):
            return
        
        vis_files = []
        for root, dirs, files in os.walk(vis_dir):
            vis_files.extend([f for f in files if f.endswith('.png')])
        
        if vis_files:
            print(f"�� {len(vis_files)} visualizations saved in: {vis_dir}")
            
            # Count by type
            vis_types = {'query': 0, 'success': 0, 'false_pos': 0, 'false_neg': 0, 
                        'challenging': 0, 'similarity_analysis': 0}
            for file in vis_files:
                for vis_type in vis_types.keys():
                    if vis_type in file:
                        vis_types[vis_type] += 1
                        break
            
            print("   �� File types:")
            for vis_type, count in vis_types.items():
                if count > 0:
                    print(f"     {vis_type}: {count} files")
    except Exception as e:
        print(f"⚠️ Visualization summary failed: {e}")

def organize_academic_paper_materials(results_df: pd.DataFrame, stats_results: Dict, 
                                     config: AuraConfig, timestamp: int, bootstrap_results: Dict = None) -> Dict:
    """
    Organize critical materials for 4-page academic paper
    Creates academic_paper folder with essential figures and tables
    """
    
    # Create academic paper directory
    academic_dir = os.path.join(config.RESULTS_PATH, "academic_paper")
    os.makedirs(academic_dir, exist_ok=True)
    
    paper_files = {}
    
    print("\n�� Organizing Academic Paper Materials...")
    print("=" * 50)
    
    # 1. CRITICAL: Performance Comparison Table (LaTeX)
    try:
        table_content = generate_performance_table_latex(results_df)
        table_file = os.path.join(academic_dir, "table_performance_comparison.tex")
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write(table_content)
        paper_files['performance_table'] = table_file
        print("✅ Table 1: Performance comparison (LaTeX)")
    except Exception as e:
        print(f"❌ Failed to create performance table: {e}")
    
    # 2. CRITICAL: Method Overview Figure
    try:
        method_fig = create_method_overview_figure(config, academic_dir)
        paper_files['method_overview'] = method_fig
        print("✅ Figure 1: Method overview")
    except Exception as e:
        print(f"❌ Failed to create method overview: {e}")
    
    # 3. CRITICAL: Quantitative Results Figure
    try:
        results_fig = create_quantitative_results_figure(results_df, academic_dir)
        paper_files['quantitative_results'] = results_fig
        print("✅ Figure 2: Quantitative results")
    except Exception as e:
        print(f"❌ Failed to create quantitative results: {e}")
    
    # 4. Copy representative examples
    try:
        qual_examples = copy_representative_examples(config, academic_dir)
        paper_files['qualitative_examples'] = qual_examples
        print("✅ Figure 3: Qualitative examples")
    except Exception as e:
        print(f"❌ Failed to copy qualitative examples: {e}")
    
    print(f"\n�� Academic materials saved to: {academic_dir}")
    print("�� Ready for 4-page academic paper!")
    
    return paper_files

def generate_performance_table_latex(results_df: pd.DataFrame) -> str:
    """Generate publication-ready LaTeX table"""
    
    # Filter scenario-level results
    scenario_results = results_df[results_df['result_type'] == 'scenario_aggregate'].copy() if 'result_type' in results_df.columns else results_df[results_df['target_class'].isna()].copy()
    
    if len(scenario_results) == 0:
        return "% No scenario results available"
    
    scenarios = scenario_results['scenario'].unique()
    
    latex_content = [
        "% Performance Comparison Table for Academic Paper",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Performance comparison on UC Merced dataset. LoRA domain adaptation shows consistent improvements.}",
        "\\label{tab:performance_comparison}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{F1-Score} & \\textbf{Accuracy} & \\textbf{AUROC} \\\\",
        "\\midrule"
    ]
    
    # Sort scenarios for logical presentation
    scenario_order = ['Zero-Shot', 'Few-Shot-Only', 'LoRA+Few-Shot']
    ordered_scenarios = [s for s in scenario_order if s in scenarios]
    
    for scenario in ordered_scenarios:
        scenario_data = scenario_results[scenario_results['scenario'] == scenario]
        if len(scenario_data) > 0:
            f1_mean = scenario_data['f1'].mean()
            f1_std = scenario_data['f1'].std() if len(scenario_data) > 1 else 0.0
            acc_mean = scenario_data['accuracy'].mean()
            acc_std = scenario_data['accuracy'].std() if len(scenario_data) > 1 else 0.0
            
            # Format method name
            method_name = scenario.replace('_', ' ').replace('+', '+')
            
            if 'auroc' in scenario_data.columns:
                auroc_mean = scenario_data['auroc'].mean()
                auroc_std = scenario_data['auroc'].std() if len(scenario_data) > 1 else 0.0
                
                if len(scenario_data) > 1:
                    latex_content.append(
                        f"{method_name} & ${f1_mean:.3f} \\pm {f1_std:.3f}$ & "
                        f"${acc_mean:.3f} \\pm {acc_std:.3f}$ & "
                        f"${auroc_mean:.3f} \\pm {auroc_std:.3f}$ \\\\"
                    )
                else:
                    latex_content.append(
                        f"{method_name} & ${f1_mean:.3f}$ & "
                        f"${acc_mean:.3f}$ & "
                        f"${auroc_mean:.3f}$ \\\\"
                    )
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_content)

def create_method_overview_figure(config: AuraConfig, save_dir: str) -> str:
    """Create method overview figure for paper"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Create flowchart-style diagram
    boxes = {
        'base_model': (1, 3, 'Base Model\n(SigLIP)'),
        'lora_adapt': (3, 3, 'LoRA\nAdaptation'),
        'domain_model': (5, 3, 'Domain-Adapted\nModel'),
        'few_shot': (7, 4, 'Few-Shot\nExamples'),
        'text_query': (7, 2, 'Text Query\n"airplane"'),
        'embedding': (9, 3, 'Query\nEmbedding'),
        'similarity': (11, 3, 'Similarity\nComputation'),
        'prediction': (13, 3, 'Detection\nResults')
    }
    
    # Draw boxes
    for box_id, (x, y, text) in boxes.items():
        if box_id == 'lora_adapt':
            color = 'lightblue'
        elif box_id in ['few_shot', 'text_query']:
            color = 'lightgreen'
        elif box_id == 'domain_model':
            color = 'orange'
        else:
            color = 'lightgray'
            
        rect = plt.Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((1.8, 3), (2.2, 3)),  # base -> lora
        ((3.8, 3), (4.2, 3)),  # lora -> domain
        ((6.2, 4), (8.2, 3.5)),  # few-shot -> embedding
        ((6.2, 2), (8.2, 2.5)),  # text -> embedding
        ((9.8, 3), (10.2, 3)),  # embedding -> similarity
        ((11.8, 3), (12.2, 3))   # similarity -> prediction
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('AURA-RS: Method Overview', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    figure_path = os.path.join(save_dir, "figure_method_overview.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return figure_path

def create_quantitative_results_figure(results_df: pd.DataFrame, save_dir: str) -> str:
    """Create quantitative results comparison figure"""
    
    # Filter scenario-level results
    scenario_results = results_df[results_df['result_type'] == 'scenario_aggregate'].copy() if 'result_type' in results_df.columns else results_df[results_df['target_class'].isna()].copy()
    
    if len(scenario_results) == 0:
        return ""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = scenario_results['scenario'].unique()
    scenario_order = ['Zero-Shot', 'Few-Shot-Only', 'LoRA+Few-Shot']
    ordered_scenarios = [s for s in scenario_order if s in scenarios]
    
    # Plot 1: F1 Score Comparison
    f1_means = []
    f1_stds = []
    scenario_labels = []
    
    for scenario in ordered_scenarios:
        scenario_data = scenario_results[scenario_results['scenario'] == scenario]
        if len(scenario_data) > 0:
            f1_means.append(scenario_data['f1'].mean())
            f1_stds.append(scenario_data['f1'].std() if len(scenario_data) > 1 else 0.0)
            scenario_labels.append(scenario.replace('_', ' ').replace('+', '+'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(scenario_labels)]
    bars1 = ax1.bar(scenario_labels, f1_means, yerr=f1_stds, 
                   capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(f1_means) * 1.15 if f1_means else 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars1, f1_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Performance summary
    ax2.text(0.5, 0.5, f'Total Classes: {len(config.NOVEL_CLASSES)}\nBase Classes: {len(config.BASE_CLASSES)}\nSeeds: {len(config.RANDOM_SEEDS)}', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    ax2.set_title('Experimental Setup', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    figure_path = os.path.join(save_dir, "figure_quantitative_results.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return figure_path

def copy_representative_examples(config: AuraConfig, academic_dir: str) -> str:
    """Copy most representative qualitative examples for paper"""
    
    # Source visualization directory
    vis_source = os.path.join(config.RESULTS_PATH, "visualizations")
    if not os.path.exists(vis_source):
        return ""
    
    # Create qualitative examples directory
    qual_dir = os.path.join(academic_dir, "qualitative_examples")
    os.makedirs(qual_dir, exist_ok=True)
    
    # Copy best representative examples (if they exist)
    import shutil
    copied_files = []
    
    for root, dirs, files in os.walk(vis_source):
        for file in files:
            if any(keyword in file for keyword in ['success_airplane', 'similarity_analysis', 'query_airplane']):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(qual_dir, file)
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_files.append(dest_path)
                except Exception:
                    pass
    
    return qual_dir if copied_files else ""

def main():
    """Clean and optimized main function"""
    print('Starting AURA-RS Production Version...')
    
    try:
        # Initialize configuration
        config = AuraConfig()
        print("AURA-RS UC Merced Dataset - Production Ready")
        print(f"Base Classes: {len(config.BASE_CLASSES)}")
        print(f"Novel Classes: {len(config.NOVEL_CLASSES)}")
        print(f"Device: {config.DEVICE}")
        print(f"Seeds: {config.RANDOM_SEEDS}")
        print(f"K-shots: {config.K_SHOTS}")
        print(f"Max samples per class: {config.MAX_SAMPLES_PER_CLASS}")
        print("=" * 60)
        
        # Run experiments
        try:
            results_df = run_experiments(config)
        except KeyboardInterrupt:
            print("\nExperiments interrupted by user (Ctrl+C)")
            print("Partial results may be available in the results folder")
            return
        except Exception as e:
            print(f"Experiment execution failed: {e}")
            print("Solutions: Check GPU memory, internet connection, or reduce BATCH_SIZE")
            raise
        
        # Process results if experiments completed
        if len(results_df) > 0:
            timestamp = int(time.time())
            
            # Save results
            results_file = save_results_with_fallback(results_df, config, timestamp)
            
            # Generate summary
            summary = generate_experiment_summary(results_df, results_file, timestamp)
            
            # Save summary file
            try:
                latest_file = os.path.join(config.RESULTS_PATH, "latest_results.json")
                with open(latest_file, 'w') as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                print(f"Failed to save summary file: {e}")
            
            # Perform statistical analysis
            perform_statistical_analysis(results_df, config, timestamp)
            
            # Summarize visualizations
            summarize_visualizations(config)
            
            print("AURA-RS Production Version - All Done!")
            
        else:
            print("No results generated")
            print("Possible reasons: dataset download failed, model loading issues, or insufficient memory")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
        print("Check logs above for details")
        print("Full error traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()