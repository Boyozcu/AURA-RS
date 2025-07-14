# AURA-RS: Advanced Unified Remote-sensing Analysis Framework

AURA-RS is a sophisticated framework for remote sensing visual grounding tasks, supporting both the DIOR-RSVG and UC Merced Land Use datasets.

## 🌟 Features

- **Dual Dataset Support**:
  - DIOR-RSVG Dataset: Remote sensing images with referring expressions
  - UC Merced Land Use Dataset: Land use classification images

- **Advanced Model Architecture**:
  - Base model: SigLIP (google/siglip-base-patch16-224)
  - LoRA-based domain adaptation
  - Efficient vision and text model integration

- **Flexible Configuration**:
  - Development mode for rapid prototyping
  - Production mode for thorough academic analysis
  - Customizable hyperparameters via AuraConfig

- **Comprehensive Tools**:
  - GPU memory management
  - Efficient image loading and processing
  - Advanced similarity computation
  - Extensive visualization capabilities

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- PIL (Python Imaging Library)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Boyozcu/AURA-RS.git
cd AURA-RS

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers peft pillow numpy pandas scikit-learn matplotlib seaborn
```

## 🚀 Quick Start

1. **Configure Your Environment**:
```python
from dior import AuraConfig

config = AuraConfig()
# For quick experiments
config.DEVELOPMENT_MODE = True
# For thorough analysis
config.DEVELOPMENT_MODE = False
```

2. **Initialize the Model**:
```python
from dior import AuraModel

model = AuraModel(config)
```

3. **Run Experiments**:
```python
# Adapt model with LoRA
model.adapt_with_lora(train_data)

# Evaluate on test data
results = model.evaluate(test_data)
```

## 📊 Available Configurations

### Development Mode
- Single seed for faster iteration
- Reduced visualizations and analysis
- ~56 experiments (3x faster)

### Production Mode
- Multiple seeds for robust statistics
- Full academic analysis and visualizations
- ~168 experiments

## 📋 Dataset Classes

- `BASE_CLASSES` (13 classes)
- `NOVEL_CLASSES` (8 classes)
- Total of 21 distinct classes for comprehensive remote sensing analysis

## 🎨 Visualization Features

- Performance plots
- Confusion matrices
- Similarity maps
- Failure analysis
- High-quality figures for academic papers

## 🔬 Academic Analysis Tools

- Statistical significance testing
- Confidence intervals
- Cross-validation
- Confusion matrix analysis
- Bootstrap sampling

## 📝 Citation

If you use AURA-RS in your research, please cite:

```bibtex
@misc{AURA-RS,
  author = {Boyozcu},
  title = {AURA-RS: Advanced Unified Remote-sensing Analysis Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Boyozcu/AURA-RS}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions and feedback, please open an issue on the GitHub repository.
