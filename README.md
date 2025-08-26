# CAS_ISIC: Classification and Segmentation System for ISIC Dataset

## Abstract

This repository contains a comprehensive deep learning system for medical image analysis, specifically designed for the International Skin Imaging Collaboration (ISIC) dataset. The system implements both classification and segmentation tasks using state-of-the-art neural network architectures, enhanced with Explainable AI (XAI) capabilities for clinical interpretability.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [XAI Implementation](#xai-implementation)
5. [File Structure](#file-structure)
6. [License](#license)

## Overview

The CAS_ISIC system is designed for automated analysis of dermoscopic images, providing both binary classification (benign/malignant) and pixel-level segmentation of skin lesions. The system incorporates multiple XAI methodologies to ensure clinical interpretability and regulatory compliance.

### Key Features

- **Dual-Model Architecture**: ResNet18 for classification, UNet++ for segmentation
- **Explainable AI Integration**: Four XAI methods (IG, Grad-CAM, SHAP, LIME) for both models
- **Graphical User Interface**: Intuitive PyQt5-based interface for clinical use
- **Modular Design**: Extensible architecture for additional models and datasets

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Windows 10/11 (primary support)

### Quick Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/karezhar/CAS_ISIC.git
   cd CAS_ISIC
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Checkpoints**
   ```bash
   # Option 1: Use batch file
   download_models.bat
   
   # Option 2: Run Python script directly
   python download_models.py
   ```

## Usage

### GUI Application

Launch the main application:

```bash
# Option 1: Use batch file
run_gui.bat

# Option 2: Run directly
python -m GUI.main
```

### XAI Mode

Enable Explainable AI analysis by toggling the "XAI Mode" button in the GUI. This generates:

- **Classification XAI**: 4 interpretability visualizations for the classification model
- **Segmentation XAI**: 4 interpretability visualizations for the segmentation model
- **Output Format**: `XAI_<image_stem>_<model>_<method>.png`

## XAI Implementation

### Implemented Methods

1. **Integrated Gradients (IG)**
   - **Purpose**: Attribution analysis
   - **Output**: Saliency maps

2. **Grad-CAM**
   - **Purpose**: Class activation mapping
   - **Output**: Heatmap overlays

3. **SHAP (SHapley Additive exPlanations)**
   - **Purpose**: Feature importance analysis
   - **Output**: Feature attribution maps

4. **LIME (Local Interpretable Model-agnostic Explanations)**
   - **Purpose**: Local model approximation
   - **Output**: Interpretable explanations

### XAI Output Structure

```
XAI_results/
├── ISIC_0000003/
│   ├── XAI_ISIC_0000003_cls_IG.png
│   ├── XAI_ISIC_0000003_cls_GradCAM.png
│   ├── XAI_ISIC_0000003_cls_SHAP.png
│   ├── XAI_ISIC_0000003_cls_LIME.png
│   ├── XAI_ISIC_0000003_segm_IG.png
│   ├── XAI_ISIC_0000003_segm_GradCAM.png
│   ├── XAI_ISIC_0000003_segm_SHAP.png
│   └── XAI_ISIC_0000003_segm_LIME.png
└── ...
```

## File Structure

### Core System Files

```
CAS_ISIC/
├── core/
│   └── pipeline.py              # Main processing pipeline
├── GUI/
│   └── main.py                  # PyQt5 GUI application
├── xai/                         # Explainable AI implementation
│   ├── __init__.py
│   ├── base.py                  # Base classes and utilities
│   ├── integrated_gradients.py  # IG implementation
│   ├── grad_cam.py             # Grad-CAM implementation
│   ├── shap_explainer.py       # SHAP implementation
│   ├── lime_explainer.py       # LIME implementation
│   └── runner.py               # XAI orchestration
├── models/                      # Model definitions
└── utils/                       # Utility functions
```

### Executable Scripts

- `run_gui.bat` - Quick GUI launcher
- `download_models.bat` - Model checkpoint downloader
- `download_models.py` - Python script for model downloads

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cas_isic_2024,
  title={CAS\_ISIC: Classification and Segmentation System for ISIC Dataset},
  author={Your Name},
  year={2024},
  url={https://github.com/karezhar/CAS_ISIC}
}
```

## Acknowledgments

- ISIC Dataset contributors
- PyTorch development team
- Captum and SHAP developers
- Medical imaging research community

---

**Note**: This system is designed for research and educational purposes. For clinical use, additional validation and regulatory compliance may be required.
