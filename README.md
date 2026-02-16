# 🩺 Hierarchical Deep Learning Framework for Diabetic Retinopathy Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clinically aligned multi-stage deep learning system for automated Diabetic Retinopathy (DR) detection and severity grading from retinal fundus images.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Clinical Motivation](#clinical-motivation)
- [System Architecture](#system-architecture)
- [Stage Details](#stage-details)
  - [Stage 0: Binary DR Detection](#stage-0-binary-dr-detection)
  - [Stage 1: Referable vs Non-Referable Classification](#stage-1-referable-vs-non-referable-classification)
  - [Stage 2: Severe vs Proliferative Classification](#stage-2-severe-vs-proliferative-classification)
- [Results](#results)
- [Technical Contributions](#technical-contributions)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Citation](#citation)

---

## 🎯 Overview

This repository implements a hierarchical deep learning framework that mimics real-world ophthalmic screening workflows for Diabetic Retinopathy detection. The system progressively refines diagnostic granularity through three stages:

```
Retinal Fundus Image
         ↓
    [Stage 0] → DR vs No DR
         ↓
    [Stage 1] → Referable vs Non-Referable
         ↓
    [Stage 2] → Severe vs Proliferative
```

**Key Features:**
- 🎯 Hierarchical classification pipeline
- ⚖️ Focal Loss for class imbalance mitigation
- 🔍 Threshold calibration for recall optimization
- 🏥 Clinical screening alignment
- 🚀 GPU-optimized training

---

## 🏥 Clinical Motivation

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness worldwide. Early detection and timely referral are critical to prevent irreversible vision loss.

### Challenges in Automated DR Detection

- **Severe class imbalance** in medical datasets
- **Subtle inter-class visual differences** between severity grades
- **Inter-patient variability** in retinal appearance
- **Threshold sensitivity** affecting recall rates
- **Overconfidence** of standard binary classifiers

### Our Solution

This project addresses these challenges through:

 **Focal Loss** - Handles class imbalance by down-weighting easy examples  
 **Class Weighting** - Balances minority class representation  
 **Progressive Fine-tuning** - Gradual unfreezing of model layers  
 **Threshold Calibration** - Optimizes decision boundaries for clinical metrics  
 **Hierarchical Classification** - Multi-stage refinement reduces error propagation

---

## 🏗️ System Architecture

### Stage 0: Binary DR Detection
**Objective:** Initial screening to separate healthy from diseased cases

- **Input:** Retinal fundus image (224×224 or 300×300)
- **Output:** No DR (0) vs DR Present (1)
- **Backbone:** EfficientNetB3 (ImageNet pretrained)

### Stage 1: Referable Classification
**Objective:** Clinical decision support for patient referral

- **Input:** Images classified as DR from Stage 0
- **Output:** Non-Referable (Mild + Moderate) vs Referable (Severe + Proliferative)
- **Backbone:** EfficientNet (PyTorch)

### Stage 2: Severity Classification
**Objective:** Fine-grained severity grading for referable cases

- **Input:** Images classified as Referable from Stage 1
- **Output:** Severe vs Proliferative DR
- **Backbone:** Fine-tuned EfficientNet

---

## 🔬 Stage Details

### Stage 0: Binary DR Detection

#### Model Configuration
```
Architecture:  EfficientNetB3
Framework:     TensorFlow/Keras
Input Size:    224×224 (primary), 300×300 (experimental)
Pooling:       Global Average Pooling
Activation:    Sigmoid
```

#### Training Strategy
- **Loss Function:** Binary Focal Loss (γ=2.0, α=0.25)
- **Optimizer:** Adam with ReduceLROnPlateau scheduler
- **Regularization:** Early stopping
- **Data Strategy:** Patient-wise split (prevents data leakage)
- **Training Approach:** Progressive fine-tuning (freeze → unfreeze)
- **Class Balancing:** Weighted loss function

#### Performance Metrics
| Metric | Before Calibration | After Calibration |
|--------|-------------------|-------------------|
| Validation Accuracy | 73-77% | **80.9%** |
| Optimal Threshold | 0.5 (default) | **0.74** |
| AUC | Improved | **Significant gain** |

#### Threshold Calibration
- Grid search performed over [0.2, 0.8]
- Optimized for clinical recall requirements
- Best threshold: **≈ 0.74**

---

### Stage 1: Referable vs Non-Referable Classification

#### Clinical Grouping
- **Non-Referable:** Mild DR + Moderate DR
- **Referable:** Severe DR + Proliferative DR

#### Model Details
- **Framework:** PyTorch
- **Loss:** CrossEntropy Loss
- **Strategy:** Fine-tuning of final layers
- **Focus:** Maximizing recall for referable cases

#### Outputs
- Confusion matrix visualization
- Classification report (precision, recall, F1)
- Per-class metrics (CSV format)
- Threshold tuning experiments

---

### Stage 2: Severe vs Proliferative Classification

#### Objective
Fine-grained classification within the referable DR category

#### Model Configuration
- **Classes:** Severe DR, Proliferative DR
- **Framework:** PyTorch
- **Loss:** CrossEntropy Loss
- **Checkpointing:** Validation-based model saving

#### Outputs
-  Stage-specific confusion matrix
-  Detailed classification metrics
-  Per-class performance analysis

---

## 📊 Results

### Directory Structure
```
results/
├── stage0_dr_detection/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── metrics.txt
│   └── roc_curve.png
│
├── stage1_referable/
│   ├── stage1_classification_report.txt
│   ├── stage1_confusion_matrix.png
│   └── stage1_metrics.csv
│
└── stage2_severity/
    ├── stageB_classification_report.txt
    ├── stageB_confusion_matrix.png
    └── stageB_metrics.csv
```

### Model Artifacts
```
models/
├── stage0_weights.h5                    # Binary DR detection weights
├── stage1_referable_model.pth           # Referable classification
├── stage2_severity_model.pth            # Severity grading
└── final_results_binary_dr_v2.json      # Aggregated results
```

---

## 🎓 Technical Contributions

This project introduces several technical innovations:

1. **Focal Loss Integration** - Addresses severe class imbalance in medical imaging
2. **Patient-wise Data Splitting** - Prevents data leakage in clinical scenarios
3. **Threshold Calibration Framework** - Optimizes recall for clinical requirements
4. **Hierarchical Classification Pipeline** - Reduces cumulative error propagation
5. **GPU-accelerated Fine-tuning** - Efficient transfer learning strategy
6. **Clinical Workflow Alignment** - Mirrors real-world screening protocols

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Clone Repository
```bash
git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

### Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Requirements File
```txt
tensorflow>=2.12.0
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=9.5.0
opencv-python>=4.8.0
```

---

## 💻 Usage

### Training Pipeline

#### Stage 0: Binary DR Detection
```python
python train_stage0.py --epochs 50 --batch_size 32 --lr 0.001
```

#### Stage 1: Referable Classification
```python
python train_stage1.py --epochs 30 --batch_size 16 --model efficientnet
```

#### Stage 2: Severity Grading
```python
python train_stage2.py --epochs 25 --batch_size 16
```

### Inference
```python
from inference import HierarchicalDRClassifier

# Load the pipeline
classifier = HierarchicalDRClassifier(
    stage0_weights='models/stage0_weights.h5',
    stage1_weights='models/stage1_referable_model.pth',
    stage2_weights='models/stage2_severity_model.pth'
)

# Predict on new image
result = classifier.predict('path/to/retinal_image.jpg')
print(f"DR Status: {result['dr_present']}")
print(f"Severity: {result['severity']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Threshold Calibration
```python
python calibrate_threshold.py --stage 0 --metric recall --target 0.95
```

---

## 📁 Project Structure

```
diabetic-retinopathy-detection/
│
├── data/                          # Dataset directory (not included)
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/                        # Trained model weights
│   ├── stage0_weights.h5
│   ├── stage1_referable_model.pth
│   └── stage2_severity_model.pth
│
├── results/                       # Training results and metrics
│   ├── stage0_dr_detection/
│   ├── stage1_referable/
│   └── stage2_severity/
│
├── src/                           # Source code
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train_stage0.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── inference.py
│   └── utils.py
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── part1_binary_dr_model.ipynb
│   └── part2_referable_binary_model.ipynb
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
```

---

## ⚠️ Limitations

- **Dataset Imbalance:** Significant class imbalance persists, especially in rare severity grades
- **Stage 1 Recall:** Recall for referable DR cases requires further improvement
- **Precision-Recall Trade-off:** Threshold tuning increases recall at the cost of precision
- **External Validation:** Model not yet evaluated on external datasets
- **Computational Cost:** Current architecture requires GPU for real-time inference
- **Interpretability:** Limited explainability features (Grad-CAM integration planned)

---

## 🔮 Future Work

### Planned Enhancements

- [ ] **External Dataset Validation** - Test on diverse population datasets
- [ ] **Explainability Integration** - Grad-CAM and attention visualization
- [ ] **Calibration Improvements** - Temperature scaling for confidence calibration
- [ ] **Multi-task Learning** - Joint training across all stages
- [ ] **Edge Deployment** - MobileNet-based lightweight models
- [ ] **Ensemble Methods** - Model averaging for improved robustness
- [ ] **Active Learning** - Uncertainty-based sample selection
- [ ] **Longitudinal Analysis** - Disease progression tracking

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 src/
black src/
```


---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{hierarchical_dr_detection,
  title={Hierarchical Deep Learning Framework for Diabetic Retinopathy Detection},
  author={ANANT RATHI},
  year={2026},
  publisher={GitHub},
  url={https://github.com/anantrathi27/diabetic-retinopathy-detection}
}
```

---

## 📧 Contact

For questions or collaborations:

- **Email:** ar5097@srmist.edu.in
- **GitHub:** [anantrathi27](https://github.com/anantrathi27)
- **LinkedIn:** [Anant Rathhi](https://www.linkedin.com/in/anant-rathi-2349a52b7/)

---

## 🙏 Acknowledgments

- **Dataset:** [Kaggle Diabetic Retinopathy Detection Challenge]
- **Pretrained Models:** EfficientNet (ImageNet weights)
- **Frameworks:** TensorFlow, PyTorch, scikit-learn community
- **Inspiration:** Clinical ophthalmology screening protocols

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for advancing automated medical diagnostics

</div>
