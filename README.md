# YOLOv5 Chicken Detection

![YOLOv5 Chicken Detection Results](https://github.com/LearnCode801/YOLOv5-Chicken-Detection/blob/main/Screenshot%202025-09-21%20121057.png)

A computer vision project implementing YOLOv5 for automated chicken detection in agricultural applications.

## Overview

This project uses YOLOv5 (You Only Look Once) deep learning model to detect chickens in images and videos. The model is custom-trained on a small dataset to demonstrate object detection capabilities for livestock monitoring and agricultural automation.

## Features

- Custom-trained YOLOv5s model for chicken detection
- Real-time inference on images and videos
- Google Colab implementation with GPU acceleration
- Comprehensive training pipeline with validation metrics
- Support for batch processing

## Technical Specifications

- **Model Architecture**: YOLOv5s (small variant)
- **Framework**: PyTorch with Ultralytics YOLOv5
- **Input Resolution**: 640x640 pixels
- **Classes**: 1 (chicken)
- **Model Size**: 7M parameters
- **Computational Cost**: 15.8 GFLOPs

## Dataset

- **Training Images**: 63 images
- **Validation Images**: 20 images
- **Total Dataset Size**: 83 images
- **Image Format**: Various resolutions, resized to 640x640
- **Annotation Format**: YOLO format with bounding boxes

## Performance Metrics

### Final Model Performance
- **Precision**: 74.0%
- **Recall**: 63.2%
- **mAP@0.5**: 67.9%
- **mAP@0.5-0.95**: 37.6%
- **Inference Speed**: ~9ms per frame (Tesla T4)

### Training Configuration
- **Epochs**: 60
- **Batch Size**: 16
- **Learning Rate**: 0.01
- **Optimizer**: SGD with momentum 0.937
- **Weight Decay**: 0.0005
- **Image Augmentations**: Enabled (HSV, rotation, scaling)

## Requirements

```python
Python 3.9+
PyTorch 2.0.0+
torchvision
ultralytics/yolov5
opencv-python
matplotlib
pillow
numpy
```

## Installation

### 1. Setup Environment
```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install dependencies
pip install -r requirements.txt
```

### 2. In Google Colab
```python
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
%pip install -qr requirements.txt

import torch
from yolov5 import utils
display = utils.notebook_init()
```

## Usage

### Training the Model

```bash
# Basic training command
python train.py --img 640 --batch 16 --epochs 60 --data path/to/dataset.yaml --weights yolov5s.pt

# Training with specific parameters
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 60 \
  --data data/custom-dataset.yaml \
  --weights yolov5s.pt \
  --project runs/train \
  --name chicken_detection
```

### Inference

#### Image Detection
```bash
# Single image
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/image.jpg --conf 0.25

# Multiple images
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/images/ --conf 0.25
```

#### Video Detection
```bash
# Video file
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/video.mp4 --conf 0.25

# Webcam (if available)
python detect.py --weights runs/train/exp/weights/best.pt --source 0 --conf 0.25
```

### Python API Usage
```python
import torch

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

# Inference
results = model('path/to/image.jpg')
results.show()  # Display results
results.save()  # Save results
```

## Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── dataset.yaml
```

### dataset.yaml
```yaml
train: path/to/images/train
val: path/to/images/val
nc: 1
names: ['chicken']
```

## Training Results

The model training progressed through distinct phases:

### Learning Phases
1. **Initial Learning (Epochs 0-10)**
   - mAP@0.5: 0.006 → 0.427
   - Model learning basic features

2. **Improvement Phase (Epochs 11-40)**  
   - mAP@0.5: 0.427 → 0.594
   - Steady performance gains

3. **Stabilization (Epochs 41-60)**
   - mAP@0.5: 0.594 → 0.679
   - Model convergence and fine-tuning

### Training Metrics Evolution
- **Box Loss**: 0.118 → 0.037 (decreased)
- **Object Loss**: 0.036 → 0.025 (decreased)  
- **Classification Loss**: 0 (single class)

## Results and Applications

### Detection Capabilities
- Successfully detects 1-4 chickens per frame
- Handles various chicken poses and orientations
- Works with different lighting conditions
- Maintains performance in group settings

### Real-world Applications
- **Livestock Monitoring**: Automated counting and tracking
- **Farm Management**: Health monitoring and behavior analysis  
- **Agricultural Automation**: Integration with feeding systems
- **Research**: Animal behavior studies

## Limitations

- **Small Dataset**: Only 83 images may limit generalization
- **Single Environment**: Training data from limited settings
- **Class Imbalance**: Only one class (chicken) detected
- **Resolution Dependency**: Optimized for 640x640 input size

## Future Improvements

1. **Dataset Expansion**: Increase training data size and diversity
2. **Multi-class Detection**: Add other farm animals
3. **Model Optimization**: Implement YOLOv8 or newer architectures
4. **Edge Deployment**: Optimize for mobile/embedded devices
5. **Data Augmentation**: Advanced augmentation techniques

## File Structure

```
project/
├── runs/
│   ├── train/
│   │   └── exp/
│   │       ├── weights/
│   │       │   ├── best.pt
│   │       │   └── last.pt
│   │       └── results.png
│   └── detect/
│       └── exp/
│           └── detected_images/
├── data/
│   └── custom-dataset.yaml
└── training_data/
    ├── images/
    └── labels/
```

## Model Weights

The trained model weights are saved as:
- `best.pt`: Best performing model during training
- `last.pt`: Final epoch weights

Model size: ~14.4MB

## Hardware Requirements

### Minimum Requirements
- GPU: CUDA-compatible (recommended)
- RAM: 8GB
- Storage: 2GB for model and dependencies

### Recommended Setup
- GPU: Tesla T4, RTX 3060 or better
- RAM: 16GB
- CUDA: 11.0+

## Citation

If you use this project, please cite:

```
YOLOv5 Chicken Detection Project
Custom implementation using Ultralytics YOLOv5
https://github.com/ultralytics/yolov5
```
