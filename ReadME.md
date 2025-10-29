# 🧠 Brain Tumor Segmentation using nnU-Net

A **deep learning-based medical imaging project** for **automatic brain tumor segmentation** from **multi-modal MRI scans** using a **3D nnU-Net architecture** with a **Flask web interface**.

---

## 📋 Overview

This project implements a complete brain tumor segmentation pipeline that:

- 🧩 Uses **nnU-Net** for precise tumor segmentation  
- 🧠 Processes **multi-modal MRI data** (T1, T1ce, T2, FLAIR)  
- 🌐 Includes a **Flask-based web interface** for visualization  
- 🎯 Supports **three tumor sub-regions**:  
  - **Whole Tumor (WT)**  
  - **Tumor Core (TC)**  
  - **Enhancing Tumor (ET)**  

---

## 🚀 Quick Start

### 🧰 Prerequisites

- Python 3.8+
- 8 GB+ RAM (16 GB recommended for training)
- 5 GB+ free disk space
- NVIDIA GPU (optional, for faster training)

---

### ⚙️ Installation & Setup

#### 1️⃣ Create & Activate Virtual Environment

**Windows**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Linux/Mac**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3️⃣ Set Up Project Structure
```bash
mkdir -p data/raw data/processed models outputs/visualizations outputs/predictions uploads
```

#### 4️⃣ Download BraTS Dataset

Place the **BraTS 2021 dataset** in the `data/raw/` directory:

```
data/raw/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_flair.nii.gz
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
├── BraTS2021_00001/
└── ...
```

---

## 🏗️ Project Structure

```
BrainTumorSegmentation/
│
├── app/                          # Flask web application
│   ├── app.py                    # Main Flask application
│   └── templates/
│       └── index.html            # Web interface
│
├── src/                          # Core Python modules
│   ├── model_nnunet.py           # 3D nnU-Net implementation
│   ├── dataset.py                # BraTS dataset loader
│   ├── data_loader.py            # DataLoader creation
│   ├── losses.py                 # Dice + CrossEntropy losses
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Model evaluation
│   ├── preprocess.py             # Data preprocessing
│   └── utils.py                  # Utilities & visualization
│
├── models/                       # Trained model checkpoints
├── data/                         # Dataset directories
│   ├── raw/                      # Original BraTS data
│   └── processed/                # Preprocessed data
├── outputs/                      # Results and predictions
│   ├── predictions/              # Model predictions
│   └── visualizations/           # Segmentation visualizations
├── uploads/                      # User-uploaded files
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## 🧠 Model Architecture

### ⚙️ nnU-Net Configuration

| Feature | Description |
|----------|-------------|
| **Input** | 4-channel 3D MRI volumes (128×128×128) |
| **Output** | 3-class segmentation masks |
| **Architecture** | 3D U-Net with instance normalization |
| **Depth** | 5 levels with skip connections |
| **Base Channels** | 32 |
| **Activation** | Leaky ReLU |
| **Normalization** | Instance Normalization |

### 🧩 Tumor Regions

| Label | Region | Description |
|--------|---------|-------------|
| **1** | Whole Tumor (WT) | Complete tumor area |
| **2** | Tumor Core (TC) | Central tumor region |
| **3** | Enhancing Tumor (ET) | Actively growing tumor area |

---

## 🛠️ Usage

### 🔄 Data Preprocessing
```bash
python src/preprocess.py
```

### 🧮 Model Training
```bash
python src/train.py
```

**Training Configuration**
| Parameter | Value |
|------------|--------|
| Epochs | 100 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Loss | Dice + CrossEntropy |
| Validation Split | 20% |

### 📊 Model Evaluation
```bash
python src/evaluate.py
```

### 🌐 Launch Web Application
```bash
python app/app.py
```

Access at 👉 **http://localhost:5000**

---

## 💻 Web Interface Features

- 📁 **Upload MRI scans** (`.nii.gz`, `.nii`, or images)
- ⚡ **Real-time segmentation**
- 🧩 **Tumor visualization** (predicted vs. ground truth)
- 📏 **Tumor volume statistics**
- 🧪 **Demo mode** with synthetic MRI data

---

## 📈 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Dice Coefficient (F1)** | Measures overlap between prediction & ground truth |
| **IoU (Jaccard)** | Intersection over Union |
| **Precision / Recall** | Accuracy of tumor detection |
| **Volume Analysis** | Quantitative tumor measurement |

---

## 🔧 Technical Details

### 🧬 Data Preprocessing
- Z-score normalization per modality  
- Resize to 128×128×128 voxels  
- Augmentation: flips, rotations, intensity scaling  
- One-hot encoding of segmentation masks  

### 📦 Key Dependencies
- **PyTorch** – Deep learning framework  
- **Flask** – Web application  
- **Nibabel** – Medical image I/O  
- **SimpleITK** – Image processing  
- **Albumentations** – Data augmentation  
- **Matplotlib** – Visualization  

---

## 📊 Results

✅ Automatic segmentation of tumor subregions  
✅ Quantitative tumor volume computation  
✅ Visual overlays for qualitative comparison  
✅ Web-based interactive interface  
✅ Performance tracking & evaluation  

---

## 🐛 Troubleshooting

| Issue | Possible Fix |
|--------|---------------|
| **Memory Errors** | Reduce batch size in `train.py` or input size |
| **Dataset Not Found** | Verify BraTS data path in `data/raw/` |
| **CUDA Out of Memory** | Enable mixed precision training or smaller model |
| **Import Errors** | Check virtual environment & reinstall dependencies |

### 🧪 Demo Mode
If BraTS dataset is unavailable:
- Synthetic MRI data is auto-generated  
- Demo predictions are shown  
- All web features are functional  

---

## 📚 References

1. **nnU-Net Paper** – Isensee et al., *Nature Methods* (2020)  
2. **BraTS Dataset** – Menze et al., *IEEE TMI* (2014)  
3. **U-Net Architecture** – Ronneberger et al., *MICCAI* (2015)

---

## 👥 Contributors

Developed as part of a **research project on medical image analysis**.

---

## 📄 License

This project is intended for **educational and research purposes only**.

> ⚠️ **Note:** The BraTS 2021 dataset requires acceptance of the official data usage agreement for research use.
