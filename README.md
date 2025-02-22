# Facial-Emotion-Recognition
## 📌 Overview
This repository contains an **advanced Facial Emotion Recognition (FER) system** using a **Multi-Scale Simplicial Transformer with Graph Attention**. The model processes facial expressions using **YOLOv8 for face detection, Graph Attention Networks (GAT), Multi-Scale Adaptive Graph (MSAG), and Multi-Scale Simplicial Transformer (MSST)** to achieve **highly accurate emotion classification**.

## 🚀 Features
✅ **YOLOv8-based Face Detection**  
✅ **Graph Attention Networks (GAT) for Feature Representation**  
✅ **Multi-Scale Adaptive Graph (MSAG) for Spatial & Channel Attention**  
✅ **Multi-Scale Simplicial Transformer (MSST) for Hierarchical Learning**  
✅ **Trained on AffectNet, FER2013, and CK+ Datasets**  

---

## 📂 Project Structure
```
Facial-Emotion-Recognition/
│── face_detection.py          # Detects and extracts faces using YOLOv8
│── graph_construction.py      # Creates a graph structure from facial features
│── graph_attention.py         # Implements Graph Attention Networks (GAT)
│── adaptive_attention.py      # Implements Hybrid Adaptive Attention (HAA)
│── msst_transformer.py        # Implements Multi-Scale Simplicial Transformer (MSST)
│── mssa_attention.py          # Implements Multi-Scale Simplicial Attention (MSSA)
│── facial_expression_model.py # Main facial emotion recognition model
│── train_model.py             # Training script for FER
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
```

---

## 📦 Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/Samia-Nawaz/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```

### **2. Create a Virtual Environment (Optional)**
```bash
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage
### **1. Face Detection**
Detect faces in an image using **YOLOv8**:
```bash
python face_detection.py
```

### **2. Train the Model**
Run the training script to train the FER model:
```bash
python train_model.py
```

### **3. Evaluate the Model**
Evaluate model performance on a test dataset:
```python
from facial_expression_model import FacialExpressionModel

# Load model
model = FacialExpressionModel(input_dim=128)

# Test with dummy data
import torch
x = torch.randn(1, 128)
output = model(x)
print("Predicted Emotion:", output)
```
